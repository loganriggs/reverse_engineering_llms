#!/usr/bin/env python3
"""Hard OOD controls matched to frozen cores for activity and write norm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_ood_cores import OUTPUT as OOD_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_matched_core_controls.json")
OUTPUT = TENSOR_ROOT / "spectral_matched_core_controls.json"
REPORT = TENSOR_ROOT / "SPECTRAL_MATCHED_CORE_CONTROLS.md"
FIGURE = TENSOR_ROOT / "spectral_matched_core_controls.png"


def stable_rng(seed: int, key: str) -> np.random.Generator:
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


def greedy_contribution_match(
    targets: torch.Tensor,
    candidates: torch.Tensor,
    contributions: torch.Tensor,
) -> torch.Tensor:
    remaining = candidates.cpu().tolist()
    selected = []
    for target in sorted(targets.cpu().tolist(), key=lambda i: float(contributions[i]), reverse=True):
        target_score = float(contributions[target])
        best = min(
            remaining,
            key=lambda i: abs(np.log(float(contributions[i]) + 1e-20) - np.log(target_score + 1e-20)),
        )
        selected.append(best)
        remaining.remove(best)
    return torch.as_tensor(selected, device="cuda", dtype=torch.long)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    cores = json.loads(CORE_OUTPUT.read_text())["cores"]
    ood = json.loads(OOD_OUTPUT.read_text())
    arrays = np.load(PILOT_ARRAYS)

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
    write_dictionary = model.blocks[1].mlp.Down.weight.float() @ decoder_original
    feature_write_norm = write_dictionary.square().sum(0).sqrt()
    records = []

    for corpus, path in config["corpora"].items():
        rows = np.load(path, mmap_mode="r")
        corpus_sites = ood["gate_selection"][corpus]["sites"]
        for category in config["categories"]:
            key = safe_name(f"semantic/seed101/{category}_raw")
            vectors = torch.from_numpy(
                arrays[f"{key}_eigenvectors"][:, : config["rank"]]
            ).cuda()
            sites = [
                site for site in corpus_sites[category]
                if site["stratum"] in ("high_active", "random_active")
            ]
            for site in sites:
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
                before = sparse_code(
                    product[0, -1], mean, std, encoder_weight, encoder_bias
                )
                after = sparse_code(
                    modified[0, -1], mean, std, encoder_weight, encoder_bias
                )
                delta_code = after - before
                contributions = delta_code.abs() * feature_write_norm
                changed = torch.nonzero(delta_code).flatten()
                own_all = torch.as_tensor(
                    cores[category][: config["core_size"]],
                    device="cuda",
                    dtype=torch.long,
                )
                own_active = own_all[delta_code[own_all] != 0]
                own_set = set(own_all.cpu().tolist())
                candidates = torch.as_tensor(
                    [int(i) for i in changed.cpu().tolist() if int(i) not in own_set],
                    device="cuda",
                    dtype=torch.long,
                )
                support_size = int(own_active.numel())
                if support_size == 0 or candidates.numel() < support_size:
                    continue
                own_write = write_dictionary[:, own_active] @ delta_code[own_active]
                own_norm = own_write.norm()

                def normalize_write(write: torch.Tensor) -> torch.Tensor:
                    return write * own_norm / write.norm().clamp_min(1e-20)

                def fidelity(write: torch.Tensor) -> float:
                    sequence_write = native_write.clone()
                    sequence_write[0, -1] += write
                    logits = suffix_from_layer1(
                        model, x0, value_bus, pre + sequence_write
                    )[0, -1]
                    value, _, _ = fisher_fidelity(
                        actual_logits,
                        actual_effect,
                        logits - native_logits,
                    )
                    return value

                matched_indices = greedy_contribution_match(
                    own_active, candidates, contributions
                )
                strongest_indices = candidates[
                    torch.argsort(contributions[candidates], descending=True)[:support_size]
                ]

                def write_for(indices: torch.Tensor) -> torch.Tensor:
                    return write_dictionary[:, indices] @ delta_code[indices]

                rng = stable_rng(
                    config["seed"],
                    f"{corpus}/{category}/{site['row']}/{site['position']}",
                )
                random_fidelities = []
                candidates_cpu = candidates.cpu().numpy()
                for _ in range(config["random_repeats"]):
                    indices = torch.as_tensor(
                        rng.choice(candidates_cpu, size=support_size, replace=False),
                        device="cuda",
                        dtype=torch.long,
                    )
                    random_fidelities.append(fidelity(normalize_write(write_for(indices))))

                records.append(
                    {
                        "corpus": corpus,
                        "category": category,
                        **site,
                        "support_size": support_size,
                        "own_write_norm": float(own_norm),
                        "methods": {
                            "own": fidelity(own_write),
                            "contribution_matched": fidelity(
                                normalize_write(write_for(matched_indices))
                            ),
                            "strongest_outside": fidelity(
                                normalize_write(write_for(strongest_indices))
                            ),
                            "random_changed_median": float(np.median(random_fidelities)),
                        },
                    }
                )

    method_names = list(records[0]["methods"])
    summary = {}
    category_summary = {}
    for corpus in config["corpora"]:
        summary[corpus] = {}
        category_summary[corpus] = {}
        for stratum in ("high_active", "random_active"):
            selected = [
                row for row in records
                if row["corpus"] == corpus and row["stratum"] == stratum
            ]
            summary[corpus][stratum] = {
                method: float(np.median([row["methods"][method] for row in selected]))
                for method in method_names
            }
            category_summary[corpus][stratum] = {}
            for category in config["categories"]:
                category_rows = [row for row in selected if row["category"] == category]
                category_summary[corpus][stratum][category] = {
                    method: float(np.median([row["methods"][method] for row in category_rows]))
                    for method in method_names
                }

    high_matched_gaps = {
        corpus: summary[corpus]["high_active"]["own"]
        - summary[corpus]["high_active"]["contribution_matched"]
        for corpus in config["corpora"]
    }
    high_random_gaps = {
        corpus: summary[corpus]["high_active"]["own"]
        - summary[corpus]["high_active"]["random_changed_median"]
        for corpus in config["corpora"]
    }
    category_wins = {
        corpus: sum(
            category_summary[corpus]["high_active"][category]["own"]
            > category_summary[corpus]["high_active"][category]["contribution_matched"]
            for category in config["categories"]
        )
        for corpus in config["corpora"]
    }
    random_active_gaps = {
        corpus: summary[corpus]["random_active"]["own"]
        - summary[corpus]["random_active"]["contribution_matched"]
        for corpus in config["corpora"]
    }
    hypotheses = {
        "M1_high_contribution_matched": {
            "supported": all(value >= 0.15 for value in high_matched_gaps.values()),
            "per_corpus": high_matched_gaps,
        },
        "M2_high_random_matched": {
            "supported": all(value >= 0.15 for value in high_random_gaps.values()),
            "per_corpus": high_random_gaps,
        },
        "M3_category_consistency": {
            "supported": all(value >= 4 for value in category_wins.values()),
            "wins_out_of_five": category_wins,
        },
        "M4_random_active_specificity": {
            "supported": all(value >= 0.10 for value in random_active_gaps.values()),
            "per_corpus": random_active_gaps,
        },
    }
    output = {
        "config": config,
        "summary": summary,
        "category_summary": category_summary,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Activity- and write-norm-matched OOD core controls",
        "",
        "Every control uses features that actually changed under the intervention, the same active support size as the own frozen core, and an exactly matched decoded residual-write norm. The contribution-matched control greedily matches individual write scores; the strongest-outside control is deliberately adversarial.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Matched controls](spectral_matched_core_controls.png)",
        "",
        "| Corpus | Stratum | Own | Contribution-matched | Strongest outside | Random changed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus in config["corpora"]:
        for stratum in ("high_active", "random_active"):
            row = summary[corpus][stratum]
            lines.append(
                f"| {corpus} | {stratum} | {row['own']:.3f} | "
                f"{row['contribution_matched']:.3f} | {row['strongest_outside']:.3f} | "
                f"{row['random_changed_median']:.3f} |"
            )
    lines += ["", "Category medians and per-example matched supports are in `spectral_matched_core_controls.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    methods = ["own", "contribution_matched", "strongest_outside", "random_changed_median"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, stratum in zip(axes, ("high_active", "random_active")):
        for corpus in config["corpora"]:
            axis.plot(
                range(len(methods)),
                [summary[corpus][stratum][method] for method in methods],
                marker="o",
                label=corpus,
            )
        axis.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
        axis.set_ylabel("median downstream fidelity")
        axis.set_title(stratum)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
