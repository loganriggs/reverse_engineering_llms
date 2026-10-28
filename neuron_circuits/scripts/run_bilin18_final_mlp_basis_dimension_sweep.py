#!/usr/bin/env python3
"""Disjoint behavioral dimension sweep for state and hybrid final-MLP bases."""

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


CONFIG = Path("configs/bilin18_final_mlp_basis_dimension_sweep.json")
OUTPUT = TENSOR_ROOT / "final_mlp_basis_dimension_sweep.json"
REPORT = TENSOR_ROOT / "FINAL_MLP_BASIS_DIMENSION_SWEEP.md"
FIGURE = TENSOR_ROOT / "final_mlp_basis_dimension_sweep.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    dimension = model.config.n_embd
    semantic = torch.from_numpy(np.load(BASIS_ARRAYS)["semantic32"]).cuda()
    discovery_states, _, _ = collect_states_and_writes(
        model,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    discovery_x = torch.from_numpy(discovery_states).cuda()
    max_dimension = max(config["dimensions"])
    _, _, state_v = torch.pca_lowrank(discovery_x, q=max_dimension, center=False, niter=5)
    bases = {}
    for requested in config["dimensions"]:
        bases[f"state{requested}"] = state_v[:, :requested]
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
            native_product = block.mlp.Left(selected_x) * block.mlp.Right(selected_x)
            native_write = block.mlp.Down(native_product) + block.mlp.Down_bias
            native_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(selected_pre + native_write, (dimension,))).float() / 30
            )
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_top = native_logits.argmax(-1)
            native_ce.extend(F.cross_entropy(native_logits, target, reduction="none").cpu().tolist())
            for name, basis in bases.items():
                projected = (selected_x @ basis) @ basis.T
                write = block.mlp.Down(block.mlp.Left(projected) * block.mlp.Right(projected)) + block.mlp.Down_bias
                logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(selected_pre + write, (dimension,))).float() / 30
                )
                log_prob = logits.log_softmax(-1)
                accumulators[name]["kl"].extend(
                    (native_probability * (native_log_prob - log_prob)).sum(-1).cpu().tolist()
                )
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

    def metric(corpus: str, family: str, size: int, key: str) -> float:
        return results[corpus]["bases"][f"{family}{size}"][key]

    hypotheses = {
        "D1_hybrid128_intermediate_fidelity": {
            "supported": all(metric(c, "hybrid", 128, "median_kl") <= 0.02 and metric(c, "hybrid", 128, "ce_increase") <= 0.03 for c in results),
            "values": {c: {"kl": metric(c, "hybrid", 128, "median_kl"), "ce": metric(c, "hybrid", 128, "ce_increase")} for c in results},
        },
        "D2_hybrid256_high_fidelity": {
            "supported": all(metric(c, "hybrid", 256, "median_kl") <= 0.01 and metric(c, "hybrid", 256, "ce_increase") <= 0.015 for c in results),
            "values": {c: {"kl": metric(c, "hybrid", 256, "median_kl"), "ce": metric(c, "hybrid", 256, "ce_increase")} for c in results},
        },
        "D3_hybrid_matched_width_advantage": {
            "supported": all(
                metric(c, "hybrid", size, "median_kl") < metric(c, "state", size, "median_kl")
                for c in results for size in config["dimensions"]
            ),
            "failures": [
                {"corpus": c, "dimension": size}
                for c in results for size in config["dimensions"]
                if metric(c, "hybrid", size, "median_kl") >= metric(c, "state", size, "median_kl")
            ],
        },
        "D4_ood_threshold_by_256": {
            "supported": all(
                any(metric(c, "hybrid", size, "median_kl") <= 0.01 for size in config["dimensions"])
                for c in results
            ),
            "minimum_dimension": {
                c: next((size for size in config["dimensions"] if metric(c, "hybrid", size, "median_kl") <= 0.01), None)
                for c in results
            },
        },
        "D5_hybrid256_top1": {
            "supported": all(metric(c, "hybrid", 256, "top1_agreement") >= 0.90 for c in results),
            "per_corpus": {c: metric(c, "hybrid", 256, "top1_agreement") for c in results},
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Behavioral dimension sweep for the final-MLP input quotient",
        "",
        "State-only and semantic+state hybrid bases are compared at matched dimensions on data disjoint from the 64D selection panel. Every replacement is propagated through final RMSNorm and the complete unembedding.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Dimension sweep](final_mlp_basis_dimension_sweep.png)",
        "",
        "| Corpus | Basis | 64 | 96 | 128 | 192 | 256 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in results:
        for family in ("state", "hybrid"):
            values = [metric(corpus, family, size, "median_kl") for size in config["dimensions"]]
            lines.append(f"| {corpus} | {family} KL | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += [
        "",
        "Dimension is input-state dimension, not operation count. These quotients still call the full native bilinear channel bank; an efficient tensor contraction is a separate requirement.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus in results:
        for family, style in (("state", "--"), ("hybrid", "-")):
            plots[0].plot(config["dimensions"], [metric(corpus, family, size, "median_kl") for size in config["dimensions"]], linestyle=style, marker="o", label=f"{corpus} {family}")
            plots[1].plot(config["dimensions"], [metric(corpus, family, size, "ce_increase") for size in config["dimensions"]], linestyle=style, marker="o", label=f"{corpus} {family}")
    plots[0].axhline(0.01, color="black", linewidth=1, linestyle=":")
    plots[0].set_yscale("log")
    plots[0].set_ylabel("median KL")
    plots[1].axhline(0.015, color="black", linewidth=1, linestyle=":")
    plots[1].set_ylabel("CE increase")
    for plot in plots:
        plot.set_xlabel("input basis dimension")
        plot.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
