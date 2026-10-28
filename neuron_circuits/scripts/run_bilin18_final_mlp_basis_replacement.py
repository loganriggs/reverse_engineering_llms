#!/usr/bin/env python3
"""End-to-end behavioral test of low-dimensional final-MLP input quotients."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_query_conditioned_interpreter import CONFIG as INTERPRETER_CONFIG
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_final_mlp_basis_replacement.json")
OUTPUT = TENSOR_ROOT / "final_mlp_basis_replacement.json"
REPORT = TENSOR_ROOT / "FINAL_MLP_BASIS_REPLACEMENT.md"
FIGURE = TENSOR_ROOT / "final_mlp_basis_replacement.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    saved = np.load(BASIS_ARRAYS)
    bases = {name: torch.from_numpy(saved[name]).cuda() for name in config["bases"]}
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    dimension = model.config.n_embd
    positions = interpreter_config["positions"]
    results = {}

    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
        accumulators = {
            name: {"kl": [], "ce": [], "agreement": [], "write_cosine": [], "relative_write_error": []}
            for name in bases
        }
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
            native_residual = selected_pre + native_write
            native_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(native_residual, (dimension,))).float() / 30
            )
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_ce.extend(F.cross_entropy(native_logits, target, reduction="none").cpu().tolist())
            native_top = native_logits.argmax(-1)
            native_write_norm = native_write.norm(dim=1).clamp_min(1e-20)
            for name, basis in bases.items():
                projected_x = (selected_x @ basis) @ basis.T
                product = block.mlp.Left(projected_x) * block.mlp.Right(projected_x)
                write = block.mlp.Down(product) + block.mlp.Down_bias
                residual = selected_pre + write
                logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(residual, (dimension,))).float() / 30
                )
                log_prob = logits.log_softmax(-1)
                kl = (native_probability * (native_log_prob - log_prob)).sum(-1)
                ce = F.cross_entropy(logits, target, reduction="none")
                cosine = F.cosine_similarity(native_write, write, dim=1)
                relative_error = (write - native_write).norm(dim=1) / native_write_norm
                accumulators[name]["kl"].extend(kl.cpu().tolist())
                accumulators[name]["ce"].extend(ce.cpu().tolist())
                accumulators[name]["agreement"].extend((logits.argmax(-1) == native_top).cpu().tolist())
                accumulators[name]["write_cosine"].extend(cosine.cpu().tolist())
                accumulators[name]["relative_write_error"].extend(relative_error.cpu().tolist())

        native_ce_mean = float(np.mean(native_ce))
        results[corpus_name] = {"native_ce": native_ce_mean, "bases": {}}
        for name, values in accumulators.items():
            results[corpus_name]["bases"][name] = {
                "median_kl": float(np.median(values["kl"])),
                "mean_kl": float(np.mean(values["kl"])),
                "ce": float(np.mean(values["ce"])),
                "ce_increase": float(np.mean(values["ce"]) - native_ce_mean),
                "top1_agreement": float(np.mean(values["agreement"])),
                "median_write_cosine": float(np.median(values["write_cosine"])),
                "median_relative_write_error": float(np.median(values["relative_write_error"])),
            }

    def row(corpus: str, basis: str) -> dict:
        return results[corpus]["bases"][basis]

    hypotheses = {
        "F1_hybrid_behavioral_fidelity": {
            "supported": all(row(c, "hybrid64")["median_kl"] <= 0.01 for c in results),
            "per_corpus": {c: row(c, "hybrid64")["median_kl"] for c in results},
        },
        "F2_hybrid_ce_fidelity": {
            "supported": all(row(c, "hybrid64")["ce_increase"] <= 0.01 for c in results),
            "per_corpus": {c: row(c, "hybrid64")["ce_increase"] for c in results},
        },
        "F3_state64_behavioral_fidelity": {
            "supported": all(
                row(c, "state64")["median_kl"] <= 0.02
                and row(c, "state64")["ce_increase"] <= 0.02 for c in results
            ),
            "per_corpus": {
                c: {"kl": row(c, "state64")["median_kl"], "ce_increase": row(c, "state64")["ce_increase"]}
                for c in results
            },
        },
        "F4_hybrid_improves_specialists": {
            "supported": all(
                row(c, "hybrid64")["median_kl"] < row(c, "semantic32")["median_kl"]
                and row(c, "hybrid64")["median_kl"] < row(c, "state64")["median_kl"]
                for c in results
            ),
            "values": {
                c: {name: row(c, name)["median_kl"] for name in ("semantic32", "state64", "hybrid64")}
                for c in results
            },
        },
        "F5_random_control_fails": {
            "supported": all(
                row(c, "random64")["median_kl"] >= 10 * row(c, "hybrid64")["median_kl"]
                for c in results
            ),
            "ratios": {c: row(c, "random64")["median_kl"] / max(row(c, "hybrid64")["median_kl"], 1e-30) for c in results},
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# End-to-end final-MLP input-basis replacement",
        "",
        "Each basis projects the actual layer-17 MLP input before recomputing all 4,608 bilinear products. The resulting write is inserted into the native final residual path, including final RMSNorm and unembedding. Metrics cover disjoint FineWeb plus OOD WikiText and TinyStories positions.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, hypothesis in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if hypothesis['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Final MLP quotient](final_mlp_basis_replacement.png)",
        "",
        "| Corpus | Basis | Median KL | CE increase | Top-1 agreement | Write cosine | Relative write error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        for name, value in corpus_row["bases"].items():
            lines.append(
                f"| {corpus} | {name} | {value['median_kl']:.4f} | {value['ce_increase']:.4f} | "
                f"{value['top1_agreement']:.1%} | {value['median_write_cosine']:.3f} | "
                f"{value['median_relative_write_error']:.3f} |"
            )
    lines += [
        "",
        "This validates an input quotient, not yet an efficient replacement: the current implementation still evaluates all 4,608 native products after projection. An efficient model must contract or factor the resulting output×basis×basis tensor and count those operations.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    names = config["bases"]
    x_positions = np.arange(len(names))
    for corpus in results:
        plots[0].plot(x_positions, [row(corpus, name)["median_kl"] for name in names], marker="o", label=corpus)
        plots[1].plot(x_positions, [row(corpus, name)["ce_increase"] for name in names], marker="o", label=corpus)
    for plot in plots:
        plot.set_xticks(x_positions, names, rotation=30, ha="right")
        plot.legend()
    plots[0].set_yscale("log")
    plots[0].set_ylabel("median native→replacement KL")
    plots[1].set_ylabel("cross-entropy increase")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
