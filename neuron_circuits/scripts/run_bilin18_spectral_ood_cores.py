#!/usr/bin/env python3
"""Validate frozen spectral companion cores on gate-selected OOD sites."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_gate_validation import collect_gate_sites
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    OUTPUT as PILOT_OUTPUT,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_ood_cores.json")
OUTPUT = TENSOR_ROOT / "spectral_ood_cores.json"
REPORT = TENSOR_ROOT / "SPECTRAL_OOD_CORES.md"
FIGURE = TENSOR_ROOT / "spectral_ood_cores.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    cores = json.loads(CORE_OUTPUT.read_text())["cores"]
    pilot = json.loads(PILOT_OUTPUT.read_text())
    arrays = np.load(PILOT_ARRAYS)
    categories = config["categories"]
    feature_indices = {
        category: pilot["spectra"][f"semantic/seed101/{category}"]["metadata"]["feature"]
        for category in categories
    }

    archive = np.load(SPARSE / "natural_large.npz")
    train = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32))
    mean = train.mean(0).cuda()
    std = train.std(0).clamp_min(1e-4).cuda()
    del train, archive
    sae = torch.load(
        SPARSE / "activation_sae_large_seed101.pt",
        map_location="cpu",
        weights_only=True,
    )
    encoder_weight = sae["encoder.weight"].float().cuda()
    encoder_bias = sae["encoder.bias"].float().cuda()
    decoder_original = std[:, None] * sae["decoder.weight"].float().cuda()
    del sae

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    down = model.blocks[1].mlp.Down.weight.float()
    write_dictionary = down @ decoder_original
    write_norm = write_dictionary.square().sum(0).sqrt()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)

    all_sites = {}
    records = []
    scan_config = {
        "positions": config["positions"],
        "stratum_size": config["stratum_size"],
        "seed": config["seed"],
    }

    for corpus, path in config["corpora"].items():
        rows = np.load(path, mmap_mode="r")
        sites, excluded, counts = collect_gate_sites(
            model,
            rows,
            categories,
            feature_indices,
            encoder_weight,
            encoder_bias,
            mean,
            std,
            scan_config,
            tokenizer,
        )
        if excluded:
            raise RuntimeError(f"OOD gate strata unavailable for {corpus}: {excluded}")
        all_sites[corpus] = {"sites": sites, "counts": counts}

        for category_index, category in enumerate(categories):
            wrong_category = categories[(category_index + 1) % len(categories)]
            key = safe_name(f"semantic/seed101/{category}_raw")
            vectors = torch.from_numpy(
                arrays[f"{key}_eigenvectors"][:, : config["rank"]]
            ).cuda()
            selected = [
                site for site in sites[category]
                if site["stratum"] in ("high_active", "random_active")
            ]
            for site in selected:
                prefix = np.array(
                    rows[site["row"], : site["position"] + 1],
                    dtype=np.int64,
                    copy=True,
                )
                ids = torch.as_tensor(prefix, device="cuda")[None]
                x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
                product = (
                    model.blocks[1].mlp.Left(normalized)
                    * model.blocks[1].mlp.Right(normalized)
                )
                native_write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
                native_logits = suffix_from_layer1(
                    model, x0, value_bus, pre + native_write
                )[0, -1]
                x = normalized[0, -1].float()
                delta = -(x @ vectors) @ vectors.T
                delta *= (
                    config["input_norm_fraction"] * x.norm()
                    / delta.norm().clamp_min(1e-20)
                )
                modified = product.clone()
                modified[0, -1] = (
                    model.blocks[1].mlp.Left(x + delta)
                    * model.blocks[1].mlp.Right(x + delta)
                )
                actual_write = model.blocks[1].mlp.Down(modified) + model.blocks[1].mlp.Down_bias
                actual_logits = suffix_from_layer1(
                    model, x0, value_bus, pre + actual_write
                )[0, -1]
                actual_effect = actual_logits - native_logits
                target_write = actual_write[0, -1] - native_write[0, -1]
                before = sparse_code(
                    product[0, -1], mean, std, encoder_weight, encoder_bias
                )
                after = sparse_code(
                    modified[0, -1], mean, std, encoder_weight, encoder_bias
                )
                delta_code = after - before
                contribution = delta_code.abs() * write_norm
                adaptive_order = torch.argsort(contribution, descending=True)

                def evaluate_write(predicted_write: torch.Tensor) -> dict:
                    predicted_sequence_write = native_write.clone()
                    predicted_sequence_write[0, -1] += predicted_write
                    predicted_logits = suffix_from_layer1(
                        model, x0, value_bus, pre + predicted_sequence_write
                    )[0, -1]
                    fidelity, cosine, _ = fisher_fidelity(
                        actual_logits,
                        actual_effect,
                        predicted_logits - native_logits,
                    )
                    return {"fidelity": fidelity, "cosine": cosine}

                def native_support(indices: list[int] | torch.Tensor) -> dict:
                    indices = torch.as_tensor(indices, device="cuda", dtype=torch.long)
                    return evaluate_write(write_dictionary[:, indices] @ delta_code[indices])

                def refit_support(indices: list[int] | torch.Tensor) -> dict:
                    indices = torch.as_tensor(indices, device="cuda", dtype=torch.long)
                    basis = write_dictionary[:, indices]
                    coefficients = torch.linalg.lstsq(basis, target_write).solution
                    return evaluate_write(basis @ coefficients)

                methods = {}
                for budget in config["budgets"]:
                    methods[f"fixed_{budget}"] = native_support(cores[category][:budget])
                    methods[f"adaptive_{budget}"] = native_support(adaptive_order[:budget])
                    methods[f"wrong_{budget}"] = native_support(cores[wrong_category][:budget])
                methods["fixed_32_refit"] = refit_support(cores[category][:32])
                methods["wrong_32_refit"] = refit_support(cores[wrong_category][:32])
                records.append(
                    {
                        "corpus": corpus,
                        "category": category,
                        "wrong_category": wrong_category,
                        **site,
                        "methods": methods,
                    }
                )

    methods = list(records[0]["methods"])
    summary = {}
    for corpus in config["corpora"]:
        summary[corpus] = {}
        for stratum in ("high_active", "random_active"):
            selected = [
                row for row in records
                if row["corpus"] == corpus and row["stratum"] == stratum
            ]
            summary[corpus][stratum] = {
                method: float(np.median([row["methods"][method]["fidelity"] for row in selected]))
                for method in methods
            }

    per_corpus = {}
    for corpus in config["corpora"]:
        high = summary[corpus]["high_active"]
        random_active = summary[corpus]["random_active"]
        per_corpus[corpus] = {
            "high_fixed_32": high["fixed_32"],
            "high_specificity_gap": high["fixed_32"] - high["wrong_32"],
            "adaptive_retention": high["fixed_32"] / max(high["adaptive_32"], 1e-30),
            "random_active_fixed_32": random_active["fixed_32"],
            "refit_specificity_gap": high["fixed_32_refit"] - high["wrong_32_refit"],
        }
    hypotheses = {
        "O1_high_active_ood_fidelity": {
            "supported": all(row["high_fixed_32"] >= 0.40 for row in per_corpus.values()),
            "per_corpus": {k: v["high_fixed_32"] for k, v in per_corpus.items()},
        },
        "O2_high_active_specificity": {
            "supported": all(row["high_specificity_gap"] >= 0.20 for row in per_corpus.values()),
            "per_corpus": {k: v["high_specificity_gap"] for k, v in per_corpus.items()},
        },
        "O3_adaptive_retention": {
            "supported": all(row["adaptive_retention"] >= 0.75 for row in per_corpus.values()),
            "per_corpus": {k: v["adaptive_retention"] for k, v in per_corpus.items()},
        },
        "O4_random_active_transfer": {
            "supported": all(row["random_active_fixed_32"] >= 0.25 for row in per_corpus.values()),
            "per_corpus": {k: v["random_active_fixed_32"] for k, v in per_corpus.items()},
        },
        "O5_refit_membership_specificity": {
            "supported": all(row["refit_specificity_gap"] >= 0.20 for row in per_corpus.values()),
            "per_corpus": {k: v["refit_specificity_gap"] for k, v in per_corpus.items()},
        },
    }
    output = {
        "config": config,
        "gate_selection": all_sites,
        "summary": summary,
        "per_corpus": per_corpus,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# OOD validation of frozen spectral companion cores",
        "",
        "Frozen FineWeb-discovered category cores are evaluated on WikiText-103 and TinyStories sites selected only by the exact SAE Top-128 gate. The intervention removes the category's rank-4 input subspace at 10% input norm.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![OOD cores](spectral_ood_cores.png)",
        "",
        "| Corpus | Stratum | Fixed 16 | Fixed 32 | Adaptive 32 | Wrong 32 | Fixed-32 refit | Wrong-32 refit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in config["corpora"]:
        for stratum in ("high_active", "random_active"):
            row = summary[corpus][stratum]
            lines.append(
                f"| {corpus} | {stratum} | {row['fixed_16']:.3f} | "
                f"{row['fixed_32']:.3f} | {row['adaptive_32']:.3f} | "
                f"{row['wrong_32']:.3f} | {row['fixed_32_refit']:.3f} | "
                f"{row['wrong_32_refit']:.3f} |"
            )
    lines += ["", "Gate counts and per-example results are in `spectral_ood_cores.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(config["corpora"])
    x = np.arange(len(corpora))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, stratum in zip(axes, ("high_active", "random_active")):
        for method in ("fixed_32", "adaptive_32", "wrong_32"):
            axis.plot(
                x,
                [summary[corpus][stratum][method] for corpus in corpora],
                marker="o",
                label=method,
            )
        axis.set_xticks(x, corpora)
        axis.set_ylabel("median downstream fidelity")
        axis.set_title(stratum)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
