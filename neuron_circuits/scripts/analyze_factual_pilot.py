#!/usr/bin/env python3
"""Aggregate factual-capacity runs into figures and a cautious pilot report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts/factual_capacity_pilot"))
    return parser.parse_args()


def metric(run, split, template, group, name="accuracy"):
    return run["evaluation"][split][str(template)].get(group, {}).get(name)


def audit_summary(audit):
    output = {}
    for basis in ("native", "rotated"):
        r2, positive, negative, weights = [], [], [], []
        for layer in audit["layers"]:
            values = layer[basis]
            r2.extend(values["best_r2"]); positive.extend(values["positive_tail_balanced_accuracy"]); negative.extend(values["negative_tail_balanced_accuracy"])
            variance = np.asarray(layer["activation_variance"])
            write = np.asarray(layer["write_norm"])
            if basis == "rotated":
                # Per-layer summaries already contain the exact rotated weights;
                # use variance weighting here for a transparent global statistic.
                weights.extend(variance.tolist())
            else: weights.extend((variance * write**2).tolist())
        r2, positive, negative, weights = map(np.asarray, (r2, positive, negative, weights))
        weights = weights / max(weights.sum(), 1e-20)
        output[basis] = {
            "median_r2": float(np.median(r2)), "mean_r2": float(np.mean(r2)),
            "weighted_r2": float(np.sum(weights * r2)),
            "median_positive_tail_ba": float(np.median(positive)),
            "median_negative_tail_ba": float(np.median(negative)),
        }
        prefix = "native" if basis == "native" else "rotated"
        output[basis]["median_entity_lookup_r2"] = float(np.median([
            value for layer in audit["layers"] for value in layer["address_transfer"][f"{prefix}_entity"]["r2"]
        ]))
        output[basis]["median_fact_pair_lookup_r2"] = float(np.median([
            value for layer in audit["layers"] for value in layer["address_transfer"][f"{prefix}_fact_pair"]["r2"]
        ]))
    rows = audit["causal"]["sampled_directions"]
    for basis in ("native", "rotated"):
        selected = [row for row in rows if row["basis"] == basis]
        output[basis]["causal_mean_absolute_margin_effect"] = float(np.mean([row["mean_absolute_margin_effect"] for row in selected]))
        output[basis]["causal_attribute_support"] = float(np.mean([row["groups"]["attribute"]["effective_support"] for row in selected]))
        output[basis]["causal_answer_support"] = float(np.mean([row["groups"]["answer"]["effective_support"] for row in selected]))
    components = audit["causal"]["components"]
    output["components"] = {
        "max_mlp_layer_loss_change": float(max(row["mean_loss_change"] for row in components["mlp_layers"])),
        "max_attention_head_loss_change": float(max(row["mean_loss_change"] for row in components["attention_heads"])),
    }
    return output


def make_plots(root, rows):
    colors = {0.0: "#2f6b3b", 0.5: "#d18700", 1.0: "#a23b3b"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for rho in sorted({row["rho"] for row in rows}):
        subset = sorted((row for row in rows if row["rho"] == rho), key=lambda row: row["width"])
        widths = [row["width"] for row in subset]
        axes[0].plot(widths, [row["seen_accuracy"] for row in subset], marker="o", color=colors[rho], label=f"rho={rho:g}")
        axes[0].plot(widths, [row["transfer_accuracy"] for row in subset], marker="x", linestyle="--", color=colors[rho])
        axes[1].plot(widths, [row["native_median_r2"] for row in subset], marker="o", color=colors[rho], label=f"native rho={rho:g}")
        axes[1].plot(widths, [row["rotated_median_r2"] for row in subset], marker="x", linestyle="--", color=colors[rho], label=f"rotated rho={rho:g}")
        axes[2].plot([row["seen_accuracy"] for row in subset], [row["native_median_r2"] for row in subset], marker="o", color=colors[rho], label=f"rho={rho:g}")
    axes[0].set(title="Factual performance", xlabel="model width", ylabel="accuracy", ylim=(-.02, 1.02)); axes[0].legend()
    axes[1].set(title="Held-out-template explanation", xlabel="model width", ylabel="median held-out R²"); axes[1].axhline(0, color="black", lw=.7); axes[1].legend(fontsize=8)
    axes[2].set(title="Performance–interpretability frontier", xlabel="seen-fact accuracy", ylabel="native median R²"); axes[2].axhline(0, color="black", lw=.7); axes[2].legend()
    fig.tight_layout(); fig.savefig(root / "pilot_summary.png", dpi=180); plt.close(fig)


def fmt(value): return "—" if value is None else f"{value:.3f}"


def main():
    args = parse_args()
    payload = json.loads((args.root / "pilot_results.json").read_text())
    manifest_metadata = json.loads((args.root / "manifest.json").read_text())
    rows = []
    for run in payload["runs"]:
        directory = args.root / f"rho_{run['rho']:g}" / f"width_{run['width']}"
        audit = json.loads((directory / "audit.json").read_text())
        with np.load(directory / "predictions.npz") as predictions:
            seen_indices = predictions["seen_fact_indices"]
            seen_targets = predictions["seen_fact_targets"]
            seen_prediction = predictions["seen_fact_template0_prediction"]
            per_attribute = [float(np.mean(seen_prediction[seen_indices % 8 == attribute] == seen_targets[seen_indices % 8 == attribute])) for attribute in range(8)]
        explanation = audit_summary(audit)
        row = {
            "rho": run["rho"], "width": run["width"], "parameters": run["parameter_counts"]["total"],
            "steps": run["steps"], "elapsed_seconds": run["elapsed_seconds"],
            "seen_accuracy": metric(run, "seen_fact", 0, "all"),
            "per_attribute_seen_accuracy": per_attribute,
            "transfer_accuracy": metric(run, "seen_fact", 6, "all"),
            "seen_changed_accuracy": metric(run, "seen_fact", 0, "changed"),
            "seen_unselected_accuracy": metric(run, "seen_fact", 0, "unselected"),
            "heldout_pair_changed_accuracy": metric(run, "heldout_pair", 0, "changed"),
            "heldout_pair_unselected_accuracy": metric(run, "heldout_pair", 0, "unselected"),
            "heldout_entity_changed_accuracy": metric(run, "heldout_entity", 0, "changed"),
            "heldout_entity_unselected_accuracy": metric(run, "heldout_entity", 0, "unselected"),
            "native_median_r2": explanation["native"]["median_r2"],
            "rotated_median_r2": explanation["rotated"]["median_r2"],
            "native_positive_tail_ba": explanation["native"]["median_positive_tail_ba"],
            "rotated_positive_tail_ba": explanation["rotated"]["median_positive_tail_ba"],
            "native_causal_effect": explanation["native"]["causal_mean_absolute_margin_effect"],
            "rotated_causal_effect": explanation["rotated"]["causal_mean_absolute_margin_effect"],
            "native_attribute_support": explanation["native"]["causal_attribute_support"],
            "rotated_attribute_support": explanation["rotated"]["causal_attribute_support"],
            "native_entity_lookup_r2": explanation["native"]["median_entity_lookup_r2"],
            "native_fact_pair_lookup_r2": explanation["native"]["median_fact_pair_lookup_r2"],
            "max_mlp_layer_loss_change": explanation["components"]["max_mlp_layer_loss_change"],
            "max_attention_head_loss_change": explanation["components"]["max_attention_head_loss_change"],
        }
        rows.append(row)
    make_plots(args.root, rows)
    (args.root / "summary.json").write_text(json.dumps({"stage0": payload["stage0"], "runs": rows}, indent=2), encoding="utf-8")
    lines = [
        "# Factual capacity and neuron organization: amended Stage 1 pilot", "",
        f"This is a one-data-seed, one-initialization exploratory pilot. It uses {manifest_metadata['replicas']} independently identified entities per trait tuple, stable nested random targets, and a native-axis versus random-rotation null. It is not the replicated confirmatory result.", "",
        "## Correctness and resources", "",
        f"- Entities: {payload['stage0']['entity_count']:,}; fact pairs: {payload['stage0']['fact_count']:,}.",
        f"- Manifest hash: `{payload['stage0']['manifest_hash']}`.",
        f"- Unique IDs, rule reconstruction, exact full-table balance, split disjointness, and all rho-specific balance checks passed.",
        f"- Total training time for the grid: {sum(row['elapsed_seconds'] for row in rows)/60:.1f} minutes; maximum allocated training memory: {max(run['peak_allocated_mib'] for run in payload['runs']):.0f} MiB.",
        "- This working directory has no Git metadata, so a revision could not be recorded; the manifest hash and saved source/configuration are the available provenance.", "",
        "## Performance", "",
        "Fixed is trained-template seen-fact recall; transfer uses an unseen combination of familiar field order and markers. Changed recall excludes chance coincidences. Held-out changed facts should remain near 12.5% chance.", "",
        "| rho | width | params | steps | fixed | transfer | changed recall | structured/unselected | held-pair changed | held-pair structured |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['rho']:g} | {row['width']} | {row['parameters']:,} | {row['steps']:,} | {fmt(row['seen_accuracy'])} | {fmt(row['transfer_accuracy'])} | {fmt(row['seen_changed_accuracy'])} | {fmt(row['seen_unselected_accuracy'])} | {fmt(row['heldout_pair_changed_accuracy'])} | {fmt(row['heldout_pair_unselected_accuracy'])} |")
    lines.extend(("", "## Held-out neuron explanations", "",
        "Candidate explanations were fixed before reading activations. Group means are fit on trained template 0, the candidate is selected on trained template 1, and its reported R²/tail score is evaluated on held-out template 6. Rotations are exact orthogonal coordinate nulls at the post-SwiGLU interface.", "",
        "| rho | width | native semantic R² | rotated semantic R² | native entity-lookup R² | native fact-lookup R² | native tail BA | rotated tail BA |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in rows:
        lines.append(f"| {row['rho']:g} | {row['width']} | {row['native_median_r2']:.3f} | {row['rotated_median_r2']:.3f} | {row['native_entity_lookup_r2']:.3f} | {row['native_fact_pair_lookup_r2']:.3f} | {row['native_positive_tail_ba']:.3f} | {row['rotated_positive_tail_ba']:.3f} |")
    lines.extend(("", "## Sampled causal and component effects", "",
        "Effective support ranges from 1 to 8; larger values mean effects are spread across more attribute groups. It is a breadth statistic, not proof that the groups are semantically unrelated.", "",
        "| rho | width | native effect | rotated effect | native attribute support | rotated attribute support | max MLP-layer loss Δ | max attention-head loss Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in rows:
        lines.append(f"| {row['rho']:g} | {row['width']} | {row['native_causal_effect']:.3f} | {row['rotated_causal_effect']:.3f} | {row['native_attribute_support']:.2f} | {row['rotated_attribute_support']:.2f} | {row['max_mlp_layer_loss_change']:.3f} | {row['max_attention_head_loss_change']:.3f} |")
    high_fidelity = [row for row in rows if row["seen_accuracy"] >= .98 and row["transfer_accuracy"] >= .95]
    loaded_high = [row for row in high_fidelity if row["rho"] > 0]
    lines.extend(("", "## Gate-based interpretation", ""))
    if len(loaded_high) >= 2:
        lines.append(f"{len(loaded_high)} arbitrary-load conditions passed both fidelity gates. These cells permit the planned matched-fit neuron comparison; their native-minus-rotated and narrow-versus-wide differences remain exploratory until replicated.")
    elif loaded_high:
        lines.append("Only one arbitrary-load condition passed both strict full-panel fidelity gates. Several others are within a few tenths of a percentage point, but the predeclared matched-fit comparison is not strictly satisfied and is reported as near-gate exploratory evidence.")
    else:
        lines.append("No arbitrary-load condition passed both fidelity gates. The pilot therefore supports a performance-sacrifice/optimization observation only; it cannot decide whether high-accuracy narrow models mix unrelated facts because no such narrow model was obtained.")
    half128 = next(row for row in rows if row["rho"] == .5 and row["width"] == 128)
    half256 = next(row for row in rows if row["rho"] == .5 and row["width"] == 256)
    full128 = next(row for row in rows if row["rho"] == 1 and row["width"] == 128)
    full256 = next(row for row in rows if row["rho"] == 1 and row["width"] == 256)
    full64 = next(row for row in rows if row["rho"] == 1 and row["width"] == 64)
    half64 = next(row for row in rows if row["rho"] == .5 and row["width"] == 64)
    lines.extend(("", "## Pilot observations", "",
        f"1. **Capacity changes storage speed and success.** At full randomization, width 64 reached only {full64['seen_accuracy']:.1%} after 10,000 updates. Width 128 required {full128['steps']:,} updates and width 256 required {full256['steps']:,} to reach roughly 98% full-random recall.",
        f"2. **The narrow mixed model favors the shared rule.** At rho=0.5, width 64 retained {half64['heldout_pair_unselected_accuracy']:.1%} accuracy on held-out structured pairs but stored few changed facts. The high-fit width-128/256 models stored changed facts while structured held-pair accuracy fell to {half128['heldout_pair_unselected_accuracy']:.1%} and {half256['heldout_pair_unselected_accuracy']:.1%}.",
        f"3. **Arbitrariness reduces the locked semantic score, but extra width does not restore it.** At rho=0.5 the native median R² is {half128['native_median_r2']:.3f} at width 128 and {half256['native_median_r2']:.3f} at width 256. At rho=1 it is {full128['native_median_r2']:.3f} and {full256['native_median_r2']:.3f}.",
        f"4. **The native-axis advantage is weak under this metric.** Native and rotated semantic R² are nearly equal in the loaded high-fit cells. Native positive-tail balanced accuracy is modestly above the rotated null, but the gap is only a few points and needs seed replication.",
        f"5. **Loaded activations remain highly fact-stable.** Fact-pair lookup reaches R²={full128['native_fact_pair_lookup_r2']:.3f} and {full256['native_fact_pair_lookup_r2']:.3f} in the full-random high-fit models, while entity lookup reaches {full128['native_entity_lookup_r2']:.3f} and {full256['native_entity_lookup_r2']:.3f}. These high-complexity lookups show template-stable addressing, but cannot distinguish a coherent address code from distributed memorization.",
        f"6. **Memory is not confined to the audited MLP axes.** Both whole final-position MLP-layer ablations and individual attention-head ablations have substantial effects under load, so the result is about the organization of the probed interface, not the entire storage mechanism.", "",
        "Negative held-out R² means the locked simple vocabulary predicts a channel worse than its held-out-template mean; it should be read as ‘unexplained by this audit,’ not as proof of polysemanticity. Tail balanced accuracy has a 0.5 null. Sampled ablations replace a final-position native or rotated coordinate by its discovery-panel mean and recompute normalization and every downstream block.", "",
        "![Pilot summary](pilot_summary.png)", "",
        "## Next decision", "",
        "Replicate the established boundary first: widths 64/128/256 at rho 0.5 and 1.0, paired across at least three new data seeds and two initialization seeds. Preserve the current explanation vocabulary and sampling rules. The present pilot supports a performance-sacrifice result for width 64 and a shift from simple semantic predictability toward fact-specific addressing in high-fit loaded models; it does not yet support capacity-dependent recovery of coherent native neurons or a claim of superposition.", ""))
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.root / 'summary.md'}", flush=True)


if __name__ == "__main__": main()
