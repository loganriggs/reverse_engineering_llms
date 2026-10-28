#!/usr/bin/env python3
"""Natural-write PCA factorization of the contracted Hybrid-64 final MLP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_hybrid64_efficient_factorization import OUTPUT as GLOBAL_OUTPUT
from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_query_conditioned_interpreter import CONFIG as INTERPRETER_CONFIG
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_hybrid64_natural_factorization.json")
OUTPUT = TENSOR_ROOT / "hybrid64_natural_factorization.json"
ARRAYS = TENSOR_ROOT / "hybrid64_natural_factorization.npz"
REPORT = TENSOR_ROOT / "HYBRID64_NATURAL_FACTORIZATION.md"
FIGURE = TENSOR_ROOT / "hybrid64_natural_factorization.png"


@torch.inference_mode()
def collect_phi(model, basis: torch.Tensor, corpus_config: dict, positions: list[int], batch_size: int):
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
    values = []
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start : start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, x = prefix_to_layer17(model, ids)
        z = x[:, positions].float().flatten(0, 1) @ basis
        values.append(torch.einsum("ni,nj->nij", z, z).flatten(1))
    return torch.cat(values)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    global_result = json.loads(GLOBAL_OUTPUT.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    model_dimension = model.config.n_embd
    basis = torch.from_numpy(np.load(BASIS_ARRAYS)[config["basis"]]).cuda()
    basis_dimension = basis.shape[1]
    left_basis = block.mlp.Left.weight.float() @ basis
    right_basis = block.mlp.Right.weight.float() @ basis
    tensor = torch.einsum("dh,hi,hj->dij", block.mlp.Down.weight.float(), left_basis, right_basis)
    tensor = 0.5 * (tensor + tensor.transpose(1, 2))
    tensor_flat = tensor.flatten(1)

    discovery_phi = collect_phi(
        model,
        basis,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    discovery_write = discovery_phi @ tensor_flat.T
    write_mean = discovery_write.mean(0)
    centered_write = discovery_write - write_mean
    _, singular, output_v = torch.pca_lowrank(
        centered_write,
        q=max(config["ranks"]),
        center=False,
        niter=5,
    )
    quadratic_factors = output_v.T @ tensor_flat
    cumulative_variance = torch.cumsum(singular.square(), 0) / singular.square().sum()

    results = {}
    positions = interpreter_config["positions"]
    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
        accumulators = {
            str(rank): {"native_kl": [], "hybrid_kl": [], "ce": [], "agreement": []}
            for rank in config["ranks"]
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

            z = selected_x @ basis
            phi = torch.einsum("ni,nj->nij", z, z).flatten(1)
            exact_write = phi @ tensor_flat.T + block.mlp.Down_bias
            exact_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(selected_pre + exact_write, (model_dimension,))).float() / 30)
            exact_log_prob = exact_logits.log_softmax(-1)
            exact_probability = exact_log_prob.exp()
            for rank in config["ranks"]:
                features = phi @ quadratic_factors[:rank].T
                projected_mean = (write_mean @ output_v[:, :rank]) @ output_v[:, :rank].T
                intercept = write_mean - projected_mean + block.mlp.Down_bias
                write = features @ output_v[:, :rank].T + intercept
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(selected_pre + write, (model_dimension,))).float() / 30)
                log_prob = logits.log_softmax(-1)
                values = accumulators[str(rank)]
                values["native_kl"].extend((native_probability * (native_log_prob - log_prob)).sum(-1).cpu().tolist())
                values["hybrid_kl"].extend((exact_probability * (exact_log_prob - log_prob)).sum(-1).cpu().tolist())
                values["ce"].extend(F.cross_entropy(logits, target, reduction="none").cpu().tolist())
                values["agreement"].extend((logits.argmax(-1) == native_top).cpu().tolist())
        native_ce_mean = float(np.mean(native_ce))
        results[corpus_name] = {"native_ce": native_ce_mean, "ranks": {}}
        for rank, values in accumulators.items():
            results[corpus_name]["ranks"][rank] = {
                "median_native_kl": float(np.median(values["native_kl"])),
                "median_additional_hybrid_kl": float(np.median(values["hybrid_kl"])),
                "ce_increase": float(np.mean(values["ce"]) - native_ce_mean),
                "native_top1_agreement": float(np.mean(values["agreement"])),
                "discovery_write_variance": float(cumulative_variance[int(rank) - 1]),
            }

    improvement = {
        corpus: {
            str(rank): 1 - results[corpus]["ranks"][str(rank)]["median_additional_hybrid_kl"]
            / global_result["results"][corpus]["ranks"][str(rank)]["median_additional_hybrid_kl"]
            for rank in config["ranks"]
        }
        for corpus in results
    }
    hypotheses = {
        "A1_natural_rank128_improves": {
            "supported": all(improvement[c]["128"] >= 0.25 for c in improvement),
            "per_corpus": {c: row["128"] for c, row in improvement.items()},
        },
        "A2_natural_rank128_preserves_hybrid": {
            "supported": all(row["ranks"]["128"]["median_additional_hybrid_kl"] <= 0.005 for row in results.values()),
            "per_corpus": {c: row["ranks"]["128"]["median_additional_hybrid_kl"] for c, row in results.items()},
        },
        "A3_natural_rank64_useful": {
            "supported": all(row["ranks"]["64"]["median_additional_hybrid_kl"] <= 0.01 for row in results.values()),
            "per_corpus": {c: row["ranks"]["64"]["median_additional_hybrid_kl"] for c, row in results.items()},
        },
        "A4_ood_transfer": {
            "supported": all(improvement[c][str(rank)] > 0 for c in ("wikitext", "tinystories") for rank in (64, 128, 256)),
            "values": {c: {str(rank): improvement[c][str(rank)] for rank in (64, 128, 256)} for c in ("wikitext", "tinystories")},
        },
        "A5_rank128_native_fidelity": {
            "supported": all(row["ranks"]["128"]["median_native_kl"] <= 0.05 for row in results.values()),
            "per_corpus": {c: row["ranks"]["128"]["median_native_kl"] for c, row in results.items()},
        },
    }
    output = {"config": config, "results": results, "improvement_over_global": improvement, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        basis=basis.cpu().numpy(),
        output_factors=output_v[:, :512].cpu().numpy(),
        quadratic_factors=quadratic_factors[:512].reshape(512, basis_dimension, basis_dimension).cpu().numpy(),
        write_mean=write_mean.cpu().numpy(),
        singular_values=singular.cpu().numpy(),
    )

    lines = [
        "# Natural-write factorization of the Hybrid-64 final MLP",
        "",
        "The exact contracted Hybrid-64 tensor is factorized using PCA of its writes on a separate FineWeb discovery panel. Each output PC has an exactly derived 64×64 quadratic scoring matrix, so execution cost matches the global tensor-SVD factorization at the same rank.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Natural factorization](hybrid64_natural_factorization.png)",
        "",
        "| Rank | Discovery write variance | FineWeb add. KL | WikiText add. KL | TinyStories add. KL | FineWeb native KL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in config["ranks"]:
        key = str(rank)
        lines.append(
            f"| {rank} | {results['fineweb']['ranks'][key]['discovery_write_variance']:.1%} | "
            f"{results['fineweb']['ranks'][key]['median_additional_hybrid_kl']:.4f} | "
            f"{results['wikitext']['ranks'][key]['median_additional_hybrid_kl']:.4f} | "
            f"{results['tinystories']['ranks'][key]['median_additional_hybrid_kl']:.4f} | "
            f"{results['fineweb']['ranks'][key]['median_native_kl']:.4f} |"
        )
    lines += [
        "",
        "This is natural-write optimal, not downstream-logit optimal. A remaining KL gap after high write variance is further evidence that output geometry and the final normalization must enter the compression objective.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, row in results.items():
        plots[0].plot(config["ranks"], [row["ranks"][str(rank)]["median_additional_hybrid_kl"] for rank in config["ranks"]], marker="o", label=corpus)
        plots[1].plot(config["ranks"], [row["ranks"][str(rank)]["median_native_kl"] for rank in config["ranks"]], marker="o", label=corpus)
    plots[0].axhline(0.005, color="black", linestyle=":", linewidth=1)
    plots[0].set_ylabel("additional KL from exact Hybrid-64")
    plots[1].axhline(0.05, color="black", linestyle=":", linewidth=1)
    plots[1].set_ylabel("median KL from native model")
    for plot in plots:
        plot.set_xscale("log", base=2)
        plot.set_yscale("log")
        plot.set_xlabel("natural output rank")
        plot.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
