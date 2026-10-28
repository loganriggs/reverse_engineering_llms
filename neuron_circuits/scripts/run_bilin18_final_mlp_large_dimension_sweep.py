#!/usr/bin/env python3
"""Large-dimension continuation of the final-MLP input-quotient sweep."""

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


CONFIG = Path("configs/bilin18_final_mlp_large_dimension_sweep.json")
OUTPUT = TENSOR_ROOT / "final_mlp_large_dimension_sweep.json"
REPORT = TENSOR_ROOT / "FINAL_MLP_LARGE_DIMENSION_SWEEP.md"
FIGURE = TENSOR_ROOT / "final_mlp_large_dimension_sweep.png"


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
    bases = {}
    for requested in config["dimensions"]:
        bases[f"state{requested}"] = state_v[:, :requested]
        if requested < model_dimension:
            hybrid_raw = torch.cat((semantic, state_v[:, : requested - semantic.shape[1]]), dim=1)
            bases[f"hybrid{requested}"] = torch.linalg.qr(hybrid_raw, mode="reduced").Q

    positions = interpreter_config["positions"]
    results = {}
    for corpus_name, corpus_config in config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
        accumulators = {name: {"kl": [], "ce": [], "agreement": []} for name in bases}
        native_ce = []
        for start in range(0, len(rows), interpreter_config["batch_size"]):
            raw = np.asarray(rows[start : start + interpreter_config["batch_size"], :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            pre, x = prefix_to_layer17(model, ids)
            selected_pre = pre[:, positions].float().flatten(0, 1)
            selected_x = x[:, positions].float().flatten(0, 1)
            target = torch.as_tensor(raw[:, np.asarray(positions) + 1].reshape(-1), device="cuda")
            native_write = block.mlp.Down(block.mlp.Left(selected_x) * block.mlp.Right(selected_x)) + block.mlp.Down_bias
            native_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(selected_pre + native_write, (model_dimension,))).float() / 30
            )
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_top = native_logits.argmax(-1)
            native_ce.extend(F.cross_entropy(native_logits, target, reduction="none").cpu().tolist())
            for name, basis in bases.items():
                projected = (selected_x @ basis) @ basis.T
                write = block.mlp.Down(block.mlp.Left(projected) * block.mlp.Right(projected)) + block.mlp.Down_bias
                logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(selected_pre + write, (model_dimension,))).float() / 30
                )
                log_prob = logits.log_softmax(-1)
                accumulators[name]["kl"].extend((native_probability * (native_log_prob - log_prob)).sum(-1).cpu().tolist())
                accumulators[name]["ce"].extend(F.cross_entropy(logits, target, reduction="none").cpu().tolist())
                accumulators[name]["agreement"].extend((logits.argmax(-1) == native_top).cpu().tolist())
        native_ce = float(np.mean(native_ce))
        results[corpus_name] = {"native_ce": native_ce, "bases": {}}
        for name, values in accumulators.items():
            results[corpus_name]["bases"][name] = {
                "median_kl": float(np.median(values["kl"])),
                "mean_kl": float(np.mean(values["kl"])),
                "ce_increase": float(np.mean(values["ce"]) - native_ce),
                "top1_agreement": float(np.mean(values["agreement"])),
            }

    def value(corpus: str, family: str, size: int, key: str) -> float:
        return results[corpus]["bases"][f"{family}{size}"][key]

    hypotheses = {
        "L1_hybrid512_high_fidelity": {
            "supported": all(value(c, "hybrid", 512, "median_kl") <= 0.01 for c in results),
            "per_corpus": {c: value(c, "hybrid", 512, "median_kl") for c in results},
        },
        "L2_hybrid768_near_exact": {
            "supported": all(value(c, "hybrid", 768, "median_kl") <= 0.003 and value(c, "hybrid", 768, "top1_agreement") >= 0.95 for c in results),
            "values": {c: {"kl": value(c, "hybrid", 768, "median_kl"), "top1": value(c, "hybrid", 768, "top1_agreement")} for c in results},
        },
        "L3_threshold_by_768": {
            "supported": all(
                any(value(c, "hybrid", size, "median_kl") <= 0.01 for size in (384, 512, 768))
                for c in results
            ),
            "minimum_dimension": {
                c: next((size for size in (384, 512, 768) if value(c, "hybrid", size, "median_kl") <= 0.01), None)
                for c in results
            },
        },
        "L4_hybrid_advantage_persists": {
            "supported": all(
                value(c, "hybrid", size, "median_kl") < value(c, "state", size, "median_kl")
                for c in results for size in (384, 512, 768, 1024)
            ),
            "failures": [
                {"corpus": c, "dimension": size}
                for c in results for size in (384, 512, 768, 1024)
                if value(c, "hybrid", size, "median_kl") >= value(c, "state", size, "median_kl")
            ],
        },
        "L5_full_basis_control": {
            "supported": all(value(c, "state", 1152, "median_kl") < 1e-6 for c in results),
            "per_corpus": {c: value(c, "state", 1152, "median_kl") for c in results},
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Large-dimension final-MLP quotient sweep",
        "",
        "The state and hybrid basis sweeps continue through the full 1,152D residual width on a second disjoint evaluation panel. This locates the behavioral tail missed by high local write R².",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Large dimension sweep](final_mlp_large_dimension_sweep.png)",
        "",
        "| Corpus | Basis | 384 | 512 | 768 | 1,024 | 1,152 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in results:
        for family in ("state", "hybrid"):
            cells = []
            for size in config["dimensions"]:
                name = f"{family}{size}"
                cells.append(f"{results[corpus]['bases'][name]['median_kl']:.5f}" if name in results[corpus]["bases"] else "—")
            lines.append(f"| {corpus} | {family} KL | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The full state basis is an algebraic control. Any residual discrepancy there is floating-point error rather than representational compression.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus in results:
        for family, style in (("state", "--"), ("hybrid", "-")):
            sizes = [size for size in config["dimensions"] if f"{family}{size}" in results[corpus]["bases"]]
            plots[0].plot(sizes, [value(corpus, family, size, "median_kl") for size in sizes], linestyle=style, marker="o", label=f"{corpus} {family}")
            plots[1].plot(sizes, [value(corpus, family, size, "top1_agreement") for size in sizes], linestyle=style, marker="o", label=f"{corpus} {family}")
    plots[0].axhline(0.01, color="black", linewidth=1, linestyle=":")
    plots[0].set_yscale("log")
    plots[0].set_ylabel("median KL")
    plots[1].axhline(0.95, color="black", linewidth=1, linestyle=":")
    plots[1].set_ylabel("native top-1 agreement")
    for plot in plots:
        plot.set_xlabel("input basis dimension")
        plot.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
