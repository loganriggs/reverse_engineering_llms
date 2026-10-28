#!/usr/bin/env python3
"""Test whether declared lexical families induce disjoint same-family behavior."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_discovery_generalization import final_logits, full_layer17_details


CONFIG = Path("configs/bilin18_cross_pair_latent_capability.json")
OUTPUT = TENSOR_ROOT / "bilin18_cross_pair_latent_capability.json"
REPORT = TENSOR_ROOT / "BILIN18_CROSS_PAIR_LATENT_CAPABILITY.md"
FIGURE = TENSOR_ROOT / "bilin18_cross_pair_latent_capability.png"


@torch.inference_mode()
def evaluate(model, raw, records, contrasts, batch_size):
    changes = []
    endpoint_energy = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        block, mixed, details = full_layer17_details(model, original, swapped)
        original_logits, _, _ = final_logits(model, block, mixed[:size].float(), details["write"][:size].float())
        swapped_logits, _, _ = final_logits(model, block, mixed[size:].float(), details["write"][size:].float())
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)[:, None]
        changes.append((direction * ((swapped_logits - original_logits) @ contrasts)).cpu())
        probabilities = original_logits.softmax(-1).double()
        delta = (swapped_logits - original_logits).double()
        delta = delta - (probabilities * delta).sum(-1, keepdim=True)
        endpoint_energy.append((probabilities * delta.square()).sum(-1).cpu())
    values = torch.cat(changes).numpy()
    correct = values[:, 0]
    random_means = values[:, 1:].mean(0)
    mean = float(correct.mean())
    return {
        "examples": len(correct),
        "win": float(np.mean(correct > 0)),
        "mean": mean,
        "median": float(np.median(correct)),
        "random_percentile": float(np.mean(random_means < mean)),
        "random_p95_mean": float(np.quantile(random_means, .95)),
        "endpoint_fisher_energy_mean": float(torch.cat(endpoint_energy).mean()),
    }


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    return float(np.corrcoef(first, second)[0, 1])


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    family_source = json.loads(Path(config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}
    families.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    results = []
    for family_index, (family, pairs) in enumerate(families.items()):
        midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrasts = make_contrasts(model.lm_head.weight.shape[0], targets, config["random_contrasts"], config["seed"] + 100 * family_index + 10 * fold, "cuda")
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
                records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + 1000 * family_index + 10 * fold + split)
                print("cross-pair", family, fold, split, len(records), flush=True)
                labels = [int(row[4]) for row in records]
                results.append({
                    "family": family, "fold": fold, "split": split,
                    "targets": len(targets), "cues": len(cues),
                    "label_counts": {str(label): labels.count(label) for label in (0, 1)},
                    **evaluate(model, raw, records, contrasts, config["batch_size"]),
                })

    grouped = {family: [row for row in results if row["family"] == family] for family in families}
    c0 = all(row["label_counts"]["0"] == row["label_counts"]["1"] and row["examples"] >= 40 for row in results)
    c1 = all(row["win"] >= .75 for row in grouped["spelling"])
    transferable = {family: all(row["win"] >= .60 for row in rows) for family, rows in grouped.items() if family != "spelling"}
    c2 = sum(transferable.values()) >= 4
    c3 = all(row["random_percentile"] >= .95 for family, rows in grouped.items() if family == "spelling" or transferable.get(family, False) for row in rows)

    terminal_family = json.loads(Path(config["terminal_family_results"]).read_text())
    terminal_spelling = json.loads(Path(config["spelling_terminal_results"]).read_text())
    mean_win = {family: float(np.mean([row["win"] for row in rows])) for family, rows in grouped.items()}
    mean_k90 = {family: float(np.mean([row["joint"]["quotient_k90_median"] for row in terminal_family["results"] if row["family"] == family])) for family in terminal_family["static_heads"]}
    mean_k90["spelling"] = float(np.mean([row["quotient_k90_median"] for row in terminal_spelling["results"] if row["family"] == "spelling"]))
    ordered = list(families)
    capability_k90_correlation = pearson([mean_win[name] for name in ordered], [mean_k90[name] for name in ordered])
    c4 = capability_k90_correlation <= -.5
    hypotheses = {
        "C0_all_cells_are_balanced": {"supported": c0},
        "C1_spelling_positive_control_transfers": {"supported": c1},
        "C2_human_semantic_families_are_transferable_variables": {"supported": c2, "families": sum(transferable.values())},
        "C3_declared_family_contrasts_beat_random_signings": {"supported": c3},
        "C4_transfer_strength_tracks_short_terminal_programs": {"supported": c4, "correlation": capability_k90_correlation},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "transferable": transferable, "family_mean_win": mean_win, "family_mean_k90": mean_k90, "results": results}, indent=2))
    lines = ["# Cross-pair latent-variable capability test", "", "A cue pair is swapped before the latest boundary, while the output contrast is built only from disjoint pairs in the same declared family. The two folds reverse cue and target sets. Spelling is the positive control. Thirty-one frozen balanced signings of the target tokens test contrast specificity.", "", "## Locked results", ""]
    for name, value in hypotheses.items():
        suffix = f" ({value['families']}/5 families)" if "families" in value else ""
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**{suffix}")
    lines += ["", f"Capability-win versus terminal-k90 correlation: `{capability_k90_correlation:+.3f}`.", "", "| Family/fold/split | N | win/mean | random percentile | endpoint energy |", "|---|---:|---:|---:|---:|"]
    for row in results: lines.append(f"| {row['family']}/{row['fold']}/{row['split']} | {row['examples']} | {row['win']:.3f}/{row['mean']:+.3g} | {row['random_percentile']:.3f} | {row['endpoint_fisher_energy_mean']:.3g} |")
    lines += ["", "## Interpretation boundary", "", "A failed family is not evidence that the model lacks sentiment, gender, tense, scale, or polarity information. It means this particular single-token edit does not establish a transferable family-level variable that predicts disjoint target-token logits at the fixed destination. Without that behavioral capability, broad or unstable terminal mechanisms cannot adjudicate how a successfully computed semantic variable would be represented."]
    REPORT.write_text("\n".join(lines) + "\n")
    names = list(families); x = np.arange(len(names)); figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(x, [mean_win[name] for name in names]); axes[0].axhline(.5, color="black", linewidth=.8); axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Mean correct shift", title="Disjoint cross-pair capability"); axes[0].tick_params(axis="x", rotation=25)
    axes[1].scatter([mean_win[name] for name in names], [mean_k90[name] for name in names]);
    for name in names: axes[1].annotate(name, (mean_win[name], mean_k90[name]), fontsize=8)
    axes[1].set(xlabel="Mean cross-pair win", ylabel="Mean terminal k90", title=f"Capability vs program length (r={capability_k90_correlation:+.2f})")
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "transferable": transferable, "family_mean_win": mean_win, "family_mean_k90": mean_k90}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
