#!/usr/bin/env python3
"""Output-conditioned spectral interpretation of the final bilinear MLP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer17_output_spectra.json")
OUTPUT = TENSOR_ROOT / "layer17_output_spectra.json"
ARRAYS = TENSOR_ROOT / "layer17_output_spectra.npz"
REPORT = TENSOR_ROOT / "LAYER17_OUTPUT_SPECTRA.md"
FIGURE = TENSOR_ROOT / "layer17_output_spectra.png"


def stable_rng(seed: int, key: str) -> np.random.Generator:
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


@torch.inference_mode()
def prefix_to_layer17(model, ids: torch.Tensor):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    for block in model.blocks[:17]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[17]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    return pre, normalized


@torch.inference_mode()
def final_logits(model, pre: torch.Tensor, mlp_input: torch.Tensor) -> torch.Tensor:
    block = model.blocks[17]
    product = block.mlp.Left(mlp_input) * block.mlp.Right(mlp_input)
    write = block.mlp.Down(product) + block.mlp.Down_bias
    residual = pre + write
    raw = model.lm_head(F.rms_norm(residual, (model.config.n_embd,))).float()
    return 30 * torch.tanh(raw / 30)


def class_contrast(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    logits = logits[..., : positive.numel()]
    return logits[..., positive].mean(-1) - logits[..., ~positive].mean(-1)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    all_flags = np.stack([token_flags(text) for text in texts])
    category_columns = {name: CATEGORIES.index(name) for name in config["categories"]}

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]

    spectra = {}
    saved_arrays = {}
    positive_masks = {}
    for category in config["categories"]:
        positive_np = all_flags[:, category_columns[category]].astype(bool)
        positive = torch.as_tensor(positive_np, device="cuda")
        positive_masks[category] = positive
        output_direction = (
            unembedding[positive].mean(0) - unembedding[~positive].mean(0)
        )
        output_direction /= output_direction.norm().clamp_min(1e-20)
        channel_coefficients = down.T @ output_direction
        form = left.T @ (channel_coefficients[:, None] * right)
        form = 0.5 * (form + form.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(form)
        order = torch.argsort(eigenvalues.abs(), descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        top_energy = float(eigenvalues[: config["rank"]].square().sum() / eigenvalues.square().sum())
        spectra[category] = {
            "positive_vocabulary_tokens": int(positive.sum()),
            "top_eigenvalues": eigenvalues[:16].cpu().tolist(),
            "top4_frobenius_energy_fraction": top_energy,
        }
        saved_arrays[f"{category}_eigenvalues"] = eigenvalues[:32].cpu().numpy()
        saved_arrays[f"{category}_eigenvectors"] = eigenvectors[:, :32].cpu().numpy()
        saved_arrays[f"{category}_low_eigenvalues"] = eigenvalues[-config["rank"] :].cpu().numpy()
        saved_arrays[f"{category}_low_eigenvectors"] = eigenvectors[:, -config["rank"] :].cpu().numpy()
        saved_arrays[f"{category}_output_direction"] = output_direction.cpu().numpy()
        del form, eigenvalues, eigenvectors
    np.savez_compressed(ARRAYS, **saved_arrays)

    scans = {}
    site_pools = {}
    positions = config["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    for corpus, corpus_config in config["corpora"].items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        scan_scores = {category: {"full": [], "top": [], "labels": []} for category in config["categories"]}
        pools = {category: {"positive": [], "negative": []} for category in config["categories"]}
        for start in range(0, len(rows), config["scan_batch_size"]):
            raw = np.asarray(rows[start : start + config["scan_batch_size"], :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            _, normalized = prefix_to_layer17(model, ids)
            selected_x = normalized[:, position_tensor].float()
            next_ids = raw[:, np.asarray(positions) + 1]
            for category in config["categories"]:
                eigenvalues = torch.from_numpy(saved_arrays[f"{category}_eigenvalues"]).cuda()
                eigenvectors = torch.from_numpy(saved_arrays[f"{category}_eigenvectors"]).cuda()
                projections = selected_x @ eigenvectors
                full_score = (
                    block.mlp.Left(selected_x)
                    * block.mlp.Right(selected_x)
                ) @ (down.T @ torch.from_numpy(saved_arrays[f"{category}_output_direction"]).cuda())
                top_score = projections[..., : config["rank"]].square() @ eigenvalues[: config["rank"]]
                labels = all_flags[next_ids, category_columns[category]].astype(bool)
                scan_scores[category]["full"].extend(full_score.flatten().cpu().tolist())
                scan_scores[category]["top"].extend(top_score.flatten().cpu().tolist())
                scan_scores[category]["labels"].extend(labels.reshape(-1).tolist())
                for flat_index, label in enumerate(labels.reshape(-1)):
                    batch_index, position_index = divmod(flat_index, len(positions))
                    row = {
                        "row": start + batch_index,
                        "position": positions[position_index],
                        "full_score": float(full_score.flatten()[flat_index]),
                        "top_score": float(top_score.flatten()[flat_index]),
                        "next_token": int(next_ids.reshape(-1)[flat_index]),
                    }
                    pools[category]["positive" if label else "negative"].append(row)
            if start % 100 == 0:
                print("layer17 scan", corpus, min(start + config["scan_batch_size"], len(rows)), "/", len(rows), flush=True)

        scans[corpus] = {}
        site_pools[corpus] = {}
        for category in config["categories"]:
            values = scan_scores[category]
            full_auc = auc(values["labels"], values["full"])
            top_auc = auc(values["labels"], values["top"])
            scans[corpus][category] = {
                "full_auc": full_auc,
                "top4_auc": top_auc,
                "positive_examples": int(np.sum(values["labels"])),
                "total_examples": len(values["labels"]),
                "eligible": int(np.sum(values["labels"])) >= 4,
                "top4_score_variance_fraction": float(
                    np.var(values["top"]) / max(np.var(values["full"]), 1e-30)
                ),
            }
            positives = sorted(
                pools[category]["positive"], key=lambda row: row["full_score"], reverse=True
            )
            negatives = pools[category]["negative"]
            size = min(config["sites_per_stratum"], len(positives))
            if size < 4:
                site_pools[corpus][category] = []
                continue
            rng = stable_rng(config["seed"], f"{corpus}/{category}")
            high_positive = positives[:size]
            remaining = positives[size:]
            random_positive = [
                remaining[index]
                for index in rng.choice(len(remaining), size=size, replace=False)
            ] if len(remaining) >= size else positives[-size:]
            random_negative = [
                negatives[index]
                for index in rng.choice(len(negatives), size=size, replace=False)
            ]
            site_pools[corpus][category] = [
                {**row, "stratum": stratum}
                for stratum, selected in (
                    ("high_positive", high_positive),
                    ("random_positive", random_positive),
                    ("random_negative", random_negative),
                )
                for row in selected
            ]

    records = []
    for corpus, corpus_config in config["corpora"].items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        for category in config["categories"]:
            eigenvectors = torch.from_numpy(saved_arrays[f"{category}_eigenvectors"]).cuda()
            top_vectors = eigenvectors[:, : config["rank"]]
            low_vectors = torch.from_numpy(
                saved_arrays[f"{category}_low_eigenvectors"]
            ).cuda()
            positive = positive_masks[category]
            by_row = {}
            for site in site_pools[corpus][category]:
                by_row.setdefault(site["row"], []).append(site)
            for row_index, sites in by_row.items():
                raw = np.asarray(rows[row_index : row_index + 1, :129], dtype=np.int64)
                ids = torch.as_tensor(raw[:, :128], device="cuda")
                pre, normalized = prefix_to_layer17(model, ids)
                for site in sites:
                    position = site["position"]
                    x = normalized[0, position].float()
                    pre_site = pre[0, position].float()
                    native_logits = final_logits(model, pre_site, x)
                    native_contrast = float(class_contrast(native_logits, positive))

                    def removal(vectors: torch.Tensor) -> torch.Tensor:
                        return -(x @ vectors) @ vectors.T

                    top_delta = removal(top_vectors)
                    top_delta *= (
                        config["input_norm_fraction"] * x.norm()
                        / top_delta.norm().clamp_min(1e-20)
                    )
                    low_delta = removal(low_vectors)
                    low_delta *= top_delta.norm() / low_delta.norm().clamp_min(1e-20)
                    variants = {}
                    for name, delta in (("top", top_delta), ("low", low_delta)):
                        changed_logits = final_logits(model, pre_site, x + delta)
                        contrast = float(class_contrast(changed_logits, positive))
                        effect = changed_logits - native_logits
                        _, _, fisher_energy = fisher_fidelity(
                            native_logits, effect, torch.zeros_like(effect)
                        )
                        variants[name] = {
                            "class_contrast_change": contrast - native_contrast,
                            "absolute_class_contrast_change": abs(contrast - native_contrast),
                            "fisher_effect_energy": fisher_energy,
                            "input_delta_norm": float(delta.norm()),
                        }
                    records.append(
                        {
                            "corpus": corpus,
                            "category": category,
                            **site,
                            "native_class_contrast": native_contrast,
                            "variants": variants,
                        }
                    )

    causal_summary = {}
    category_causal = {}
    for corpus in config["corpora"]:
        causal_summary[corpus] = {}
        category_causal[corpus] = {}
        for stratum in ("high_positive", "random_positive", "random_negative"):
            selected = [
                row for row in records
                if row["corpus"] == corpus and row["stratum"] == stratum
            ]
            causal_summary[corpus][stratum] = {
                variant: float(np.median([
                    row["variants"][variant]["absolute_class_contrast_change"]
                    for row in selected
                ]))
                for variant in ("top", "low")
            }
            category_causal[corpus][stratum] = {}
            for category in config["categories"]:
                category_rows = [row for row in selected if row["category"] == category]
                category_causal[corpus][stratum][category] = (
                    {
                        variant: float(np.median([
                            row["variants"][variant]["absolute_class_contrast_change"]
                            for row in category_rows
                        ]))
                        for variant in ("top", "low")
                    }
                    if category_rows else None
                )

    semantic_medians = {
        corpus: {
            "full": float(np.median([
                row["full_auc"] for row in scans[corpus].values() if row["eligible"]
            ])),
            "top4": float(np.median([
                row["top4_auc"] for row in scans[corpus].values() if row["eligible"]
            ])),
        }
        for corpus in config["corpora"]
    }
    high_ratios = {
        corpus: causal_summary[corpus]["high_positive"]["top"]
        / max(causal_summary[corpus]["high_positive"]["low"], 1e-30)
        for corpus in config["corpora"]
    }
    random_ratios = {
        corpus: causal_summary[corpus]["random_positive"]["top"]
        / max(causal_summary[corpus]["random_positive"]["low"], 1e-30)
        for corpus in config["corpora"]
    }
    category_wins = {
        corpus: sum(
            row is not None and row["top"] > row["low"]
            for row in category_causal[corpus]["high_positive"].values()
        )
        for corpus in config["corpora"]
    }
    eligible_causal_categories = {
        corpus: sum(
            row is not None
            for row in category_causal[corpus]["high_positive"].values()
        )
        for corpus in config["corpora"]
    }
    ood_auc_counts = {
        corpus: sum(
            row["eligible"] and row["top4_auc"] >= 0.60
            for row in scans[corpus].values()
        )
        for corpus in ("wikitext103", "tinystories")
    }
    hypotheses = {
        "L1_compact_semantic_readout": {
            "supported": all(
                row["top4"] >= 0.60 and row["top4"] >= row["full"] - 0.05
                for row in semantic_medians.values()
            ),
            "per_corpus": semantic_medians,
        },
        "L2_high_positive_causal_specificity": {
            "supported": all(value >= 2.0 for value in high_ratios.values()),
            "per_corpus_ratio": high_ratios,
        },
        "L3_random_positive_causal_specificity": {
            "supported": all(value >= 1.5 for value in random_ratios.values()),
            "per_corpus_ratio": random_ratios,
        },
        "L4_category_consistency": {
            "supported": all(
                category_wins[corpus] >= 4 and eligible_causal_categories[corpus] >= 4
                for corpus in config["corpora"]
            ),
            "wins_out_of_five": category_wins,
            "eligible_categories": eligible_causal_categories,
        },
        "L5_distribution_transfer": {
            "supported": all(value >= 4 for value in ood_auc_counts.values()),
            "categories_above_0.60": ood_auc_counts,
        },
    }
    output = {
        "config": config,
        "spectra": spectra,
        "scans": scans,
        "site_pools": site_pools,
        "causal_summary": causal_summary,
        "category_causal": category_causal,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 output-conditioned spectral pilot",
        "",
        "For each output token class, this constructs the exact pre-final-norm quadratic form of the final bilinear MLP along that class's unembedding contrast. It evaluates rank-4 semantic prediction and finite input-subspace interventions on FineWeb, WikiText-103, and TinyStories.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Layer-17 output spectra](layer17_output_spectra.png)",
        "",
        "| Corpus | Median full-form AUROC | Median top-4 AUROC | High-positive top/low effect | Random-positive top/low effect |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus in config["corpora"]:
        lines.append(
            f"| {corpus} | {semantic_medians[corpus]['full']:.3f} | "
            f"{semantic_medians[corpus]['top4']:.3f} | {high_ratios[corpus]:.2f}× | "
            f"{random_ratios[corpus]:.2f}× |"
        )
    lines += [
        "",
        "The quadratic form is exact only before the final residual addition, RMSNorm, and logit tanh. The finite intervention recomputes all of those operations. Per-category AUROCs and effects are in `layer17_output_spectra.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(config["corpora"])
    x = np.arange(len(corpora))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(x, [semantic_medians[c]["full"] for c in corpora], marker="o", label="full form")
    axes[0].plot(x, [semantic_medians[c]["top4"] for c in corpora], marker="o", label="top 4")
    axes[0].set_xticks(x, corpora)
    axes[0].set_ylabel("median next-token AUROC")
    axes[0].legend()
    axes[1].plot(x, [high_ratios[c] for c in corpora], marker="o", label="high positive")
    axes[1].plot(x, [random_ratios[c] for c in corpora], marker="o", label="random positive")
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("top / low absolute class-logit effect")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
