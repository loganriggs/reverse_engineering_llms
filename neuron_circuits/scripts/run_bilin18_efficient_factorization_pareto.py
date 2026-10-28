#!/usr/bin/env python3
"""Map the efficient input-dimension/output-rank frontier for the final MLP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_query_conditioned_interpreter import CONFIG as INTERPRETER_CONFIG, collect_states_and_writes
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_efficient_factorization_pareto.json")
OUTPUT = TENSOR_ROOT / "efficient_factorization_pareto.json"
REPORT = TENSOR_ROOT / "EFFICIENT_FACTORIZATION_PARETO.md"
FIGURE = TENSOR_ROOT / "efficient_factorization_pareto.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    model_dimension = model.config.n_embd
    channels = block.mlp.Left.weight.shape[0]
    native_multiplications = 3 * model_dimension * channels
    semantic = torch.from_numpy(np.load(BASIS_ARRAYS)["semantic32"]).cuda()
    discovery_states, _, _ = collect_states_and_writes(
        model,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    discovery_x = torch.from_numpy(discovery_states).cuda()
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    state_v = state_vh.T

    programs = {}
    for input_dimension in config["input_dimensions"]:
        if input_dimension == 32:
            basis = semantic
        else:
            basis = torch.linalg.qr(
                torch.cat((semantic, state_v[:, : input_dimension - 32]), dim=1), mode="reduced"
            ).Q
        projected_x = (discovery_x @ basis) @ basis.T
        discovery_write = block.mlp.Down(
            block.mlp.Left(projected_x) * block.mlp.Right(projected_x)
        )
        write_mean = discovery_write.mean(0)
        centered = discovery_write - write_mean
        _, _, output_v = torch.pca_lowrank(
            centered,
            q=max(config["output_ranks"]),
            center=False,
            niter=5,
        )
        left_basis = block.mlp.Left.weight.float() @ basis
        right_basis = block.mlp.Right.weight.float() @ basis
        channel_coefficients = output_v.T @ block.mlp.Down.weight.float()
        quadratic = torch.einsum("kh,hi,hj->kij", channel_coefficients, left_basis, right_basis)
        quadratic = 0.5 * (quadratic + quadratic.transpose(1, 2))
        programs[str(input_dimension)] = {
            "basis": basis,
            "output": output_v,
            "quadratic": quadratic.flatten(1),
            "mean": write_mean,
        }

    compute = {}
    for input_dimension in config["input_dimensions"]:
        compute[str(input_dimension)] = {}
        for output_rank in config["output_ranks"]:
            multiplications = (
                model_dimension * input_dimension
                + input_dimension * input_dimension
                + output_rank * input_dimension * input_dimension
                + output_rank * model_dimension
            )
            compute[str(input_dimension)][str(output_rank)] = {
                "multiplications_per_token": multiplications,
                "speedup": native_multiplications / multiplications,
                "parameters": (
                    model_dimension * input_dimension
                    + output_rank * input_dimension * input_dimension
                    + output_rank * model_dimension
                    + model_dimension
                ),
            }

    results = {}
    positions = interpreter_config["positions"]
    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
        accumulators = {
            f"{input_dimension}x{output_rank}": {"kl": [], "ce": [], "agreement": []}
            for input_dimension in config["input_dimensions"]
            for output_rank in config["output_ranks"]
        }
        native_ce = []
        for start in range(0, len(rows), interpreter_config["batch_size"]):
            raw = np.asarray(rows[start : start + interpreter_config["batch_size"], :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            pre, x = prefix_to_layer17(model, ids)
            selected_pre = pre[:, positions].float().flatten(0, 1)
            selected_x = x[:, positions].float().flatten(0, 1)
            target = torch.as_tensor(raw[:, np.asarray(positions) + 1].reshape(-1), device="cuda")
            native_write = block.mlp.Down(block.mlp.Left(selected_x) * block.mlp.Right(selected_x)) + block.mlp.Down_bias
            native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(selected_pre + native_write, (model_dimension,))).float() / 30)
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_top = native_logits.argmax(-1)
            native_ce.extend(F.cross_entropy(native_logits, target, reduction="none").cpu().tolist())
            for input_dimension in config["input_dimensions"]:
                program = programs[str(input_dimension)]
                z = selected_x @ program["basis"]
                phi = torch.einsum("ni,nj->nij", z, z).flatten(1)
                all_features = phi @ program["quadratic"].T
                for output_rank in config["output_ranks"]:
                    output = program["output"][:, :output_rank]
                    projected_mean = (program["mean"] @ output) @ output.T
                    intercept = program["mean"] - projected_mean + block.mlp.Down_bias
                    write = all_features[:, :output_rank] @ output.T + intercept
                    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(selected_pre + write, (model_dimension,))).float() / 30)
                    log_prob = logits.log_softmax(-1)
                    values = accumulators[f"{input_dimension}x{output_rank}"]
                    values["kl"].extend((native_probability * (native_log_prob - log_prob)).sum(-1).cpu().tolist())
                    values["ce"].extend(F.cross_entropy(logits, target, reduction="none").cpu().tolist())
                    values["agreement"].extend((logits.argmax(-1) == native_top).cpu().tolist())
        native_ce_mean = float(np.mean(native_ce))
        results[corpus_name] = {"native_ce": native_ce_mean, "programs": {}}
        for name, values in accumulators.items():
            results[corpus_name]["programs"][name] = {
                "median_native_kl": float(np.median(values["kl"])),
                "ce_increase": float(np.mean(values["ce"]) - native_ce_mean),
                "native_top1_agreement": float(np.mean(values["agreement"])),
            }

    def kl(corpus: str, input_dimension: int, output_rank: int) -> float:
        return results[corpus]["programs"][f"{input_dimension}x{output_rank}"]["median_native_kl"]

    hypotheses = {
        "P1_128x64_useful_frontier": {
            "supported": all(kl(c, 128, 64) <= 0.04 for c in results) and compute["128"]["64"]["speedup"] >= 10,
            "per_corpus": {c: kl(c, 128, 64) for c in results},
            "speedup": compute["128"]["64"]["speedup"],
        },
        "P2_128x128_higher_fidelity": {
            "supported": all(kl(c, 128, 128) <= 0.035 for c in results),
            "per_corpus": {c: kl(c, 128, 128) for c in results},
        },
        "P3_256x64_strong_fidelity": {
            "supported": all(kl(c, 256, 64) <= 0.03 for c in results),
            "per_corpus": {c: kl(c, 256, 64) for c in results},
            "speedup": compute["256"]["64"]["speedup"],
        },
        "P4_monotone_input_width": {
            "supported": all(
                all(kl(c, dimensions[index + 1], rank) <= kl(c, dimensions[index], rank) for index in range(len(dimensions) - 1))
                for c in results for rank in config["output_ranks"]
                for dimensions in [config["input_dimensions"]]
            ),
            "violations": [
                {"corpus": c, "rank": rank, "before": config["input_dimensions"][index], "after": config["input_dimensions"][index + 1]}
                for c in results for rank in config["output_ranks"] for index in range(len(config["input_dimensions"]) - 1)
                if kl(c, config["input_dimensions"][index + 1], rank) > kl(c, config["input_dimensions"][index], rank)
            ],
        },
        "P5_monotone_output_rank": {
            "supported": all(
                all(kl(c, dimension, ranks[index + 1]) <= kl(c, dimension, ranks[index]) for index in range(len(ranks) - 1))
                for c in results for dimension in config["input_dimensions"]
                for ranks in [config["output_ranks"]]
            ),
            "violations": [
                {"corpus": c, "dimension": dimension, "before": config["output_ranks"][index], "after": config["output_ranks"][index + 1]}
                for c in results for dimension in config["input_dimensions"] for index in range(len(config["output_ranks"]) - 1)
                if kl(c, dimension, config["output_ranks"][index + 1]) > kl(c, dimension, config["output_ranks"][index])
            ],
        },
    }
    output = {"config": config, "compute": compute, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Efficient final-MLP compute–fidelity frontier",
        "",
        "Hybrid input dimensions 32/64/128/256 are crossed with natural-write output ranks 32/64/128. Every cell is an executable quadratic program without native-channel scoring; multiply counts include input projection, monomials, quadratic scoring, and output decoding.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Efficient Pareto](efficient_factorization_pareto.png)",
        "",
        "| Input×output | Speedup | FineWeb KL | WikiText KL | TinyStories KL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for input_dimension in config["input_dimensions"]:
        for output_rank in config["output_ranks"]:
            lines.append(
                f"| {input_dimension}×{output_rank} | {compute[str(input_dimension)][str(output_rank)]['speedup']:.1f}× | "
                f"{kl('fineweb', input_dimension, output_rank):.4f} | {kl('wikitext', input_dimension, output_rank):.4f} | "
                f"{kl('tinystories', input_dimension, output_rank):.4f} |"
            )
    lines += [
        "",
        "These are algorithmic dense-multiply counts, not wall-clock benchmarks. The unembedding dominates neither count here nor the comparison because it is unchanged across models.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plot = plt.subplots(figsize=(8, 5))
    markers = {"fineweb": "o", "wikitext": "s", "tinystories": "^"}
    for corpus in results:
        for input_dimension in config["input_dimensions"]:
            xs, ys = [], []
            for output_rank in config["output_ranks"]:
                xs.append(compute[str(input_dimension)][str(output_rank)]["speedup"])
                ys.append(kl(corpus, input_dimension, output_rank))
            plot.plot(xs, ys, marker=markers[corpus], label=f"{corpus} input {input_dimension}")
    plot.set_xscale("log")
    plot.set_yscale("log")
    plot.invert_xaxis()
    plot.set_xlabel("algorithmic speedup over native dense MLP")
    plot.set_ylabel("median native-model KL")
    plot.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
