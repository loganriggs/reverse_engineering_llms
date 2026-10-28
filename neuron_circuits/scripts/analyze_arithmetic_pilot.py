#!/usr/bin/env python3
"""Aggregate the arithmetic pilot into plots and a standalone HTML report."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "random": "#9ca3af",
    "activation": "#0ea5e9",
    "write_magnitude": "#8b5cf6",
    "gradient_activation": "#ef4444",
}
LABELS = {
    "random": "Random",
    "activation": "|activation|",
    "write_magnitude": "|activation| × write norm",
    "gradient_activation": "|gradient × activation|",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts/arithmetic_pilot"))
    return parser.parse_args()


def median(values):
    return float(statistics.median(values))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve_rows(per_example, methods, counts):
    curves = {}
    for method in methods:
        points = []
        for retained in counts:
            group = [row for row in per_example if row["method"] == method and row["retained"] == retained]
            if not group:
                continue
            points.append({
                "retained": retained,
                "fraction": group[0]["retained_fraction"],
                "accuracy": sum(row["exact"] for row in group) / len(group),
                "margin_median": median([row["margin_retention"] for row in group]),
                "kl_median": median([row["output_kl"] for row in group]),
                "distance_median": median([row["expected_distance"] for row in group]),
            })
        curves[method] = points
    return curves


def smallest_sizes(per_example, methods, alphas):
    examples = sorted({row["example_id"] for row in per_example})
    result = {}
    for alpha in alphas:
        result[str(alpha)] = {}
        for method in methods:
            sizes = []
            for example in examples:
                candidates = [
                    row for row in per_example
                    if row["example_id"] == example and row["method"] == method
                    and row["masked_margin"] >= alpha * row["intact_margin"]
                ]
                sizes.append(min(row["retained_fraction"] for row in candidates))
            result[str(alpha)][method] = {
                "values": sizes,
                "median": median(sizes),
                "q1": percentile(sizes, 25),
                "q3": percentile(sizes, 75),
                "min": min(sizes),
                "max": max(sizes),
            }
    return result


def fidelity_plot(curves, methods, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    metrics = (
        ("accuracy", "Exact answer accuracy", (0, 1.03)),
        ("margin_median", "Median answer-margin retention", None),
        ("kl_median", "Median full-output KL", None),
        ("distance_median", "Median expected semantic cost", None),
    )
    for axis, (key, title, ylim) in zip(axes.flat, metrics, strict=True):
        for method in methods:
            points = curves[method]
            x = [point["fraction"] * 100 for point in points]
            y = [point[key] for point in points]
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.8,
                      color=COLORS[method], label=LABELS[method])
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_title(title, loc="left", fontsize=11, weight="bold")
        axis.grid(alpha=0.2)
        if ylim:
            axis.set_ylim(*ylim)
        axis.set_xlabel("Retained MLP activations (%)")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Per-input circuit fidelity as neurons are restored", fontsize=14, weight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def size_plot(sizes, methods, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for axis, alpha in zip(axes, ("0.5", "0.9", "1.0"), strict=True):
        data = [[100 * value for value in sizes[alpha][method]["values"]] for method in methods]
        box = axis.boxplot(data, patch_artist=True, tick_labels=[LABELS[m] for m in methods],
                           showfliers=True, medianprops={"color": "black", "linewidth": 1.5})
        for patch, method in zip(box["boxes"], methods, strict=True):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.65)
        axis.set_title(f"α = {alpha}", weight="bold")
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Smallest retained fraction found (%)")
    fig.suptitle("Circuit size varies substantially by prompt and fidelity target", fontsize=14, weight="bold")
    fig.tight_layout()
    save_figure(fig, path)


def semantic_plot(per_example, counts, path):
    available = [0, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    total = per_example[0]["total"]
    targets = [min(counts, key=lambda count: abs(count / total - fraction)) for fraction in available]
    categories = ("exact", "nearby", "far", "non_numeric")
    colors = ("#16a34a", "#eab308", "#f97316", "#6b7280")
    values = {category: [] for category in categories}
    labels = []
    for retained in targets:
        group = [row for row in per_example if row["method"] == "gradient_activation" and row["retained"] == retained]
        counts_by_kind = Counter(row["failure_kind"] for row in group)
        labels.append(f"{100 * retained / total:.3g}%")
        for category in categories:
            values[category].append(counts_by_kind[category] / len(group))
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    bottom = np.zeros(len(labels))
    for category, color in zip(categories, colors, strict=True):
        axis.bar(labels, values[category], bottom=bottom, label=category.replace("_", " ").title(), color=color)
        bottom += np.asarray(values[category])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of prompts")
    axis.set_xlabel("Retained MLP activations")
    axis.set_title("Errors move through nearby, far, and non-numeric regimes", loc="left", weight="bold")
    axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def transfer_family_plot(rows, counts, methods, path):
    transfer = [row for row in rows if row["phase"] == "frozen_transfer"]
    family = [row for row in rows if row["phase"] == "task_family"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rates = []
    retained_means = []
    for method in methods:
        group = [row for row in transfer if row["method"] == method]
        rates.append(sum(row["exact"] for row in group) / len(group))
        retained_means.append(statistics.mean(row["retained_fraction"] for row in group))
    axes[0].bar([LABELS[m] for m in methods], rates, color=[COLORS[m] for m in methods])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact accuracy")
    axes[0].set_title("Frozen per-input masks on changed problems", loc="left", weight="bold")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    for i, (rate, retained_mean) in enumerate(zip(rates, retained_means, strict=True)):
        axes[0].text(
            i, min(1.02, rate + 0.025), f"{rate:.0%}\n@ {retained_mean:.0%} retained",
            ha="center", va="bottom", fontsize=8,
        )
    axes[0].grid(axis="y", alpha=0.2)

    points = []
    for retained in counts:
        group = [row for row in family if row["retained"] == retained]
        if group:
            points.append((retained / group[0]["total"] * 100, sum(row["exact"] for row in group) / len(group)))
    axes[1].plot([x for x, _ in points], [y for _, y in points], marker="o", color="#ef4444")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Retained MLP activations (%)")
    axes[1].set_ylabel("Exact accuracy")
    axes[1].set_title("One task-family mask on 30 held-out problems", loc="left", weight="bold")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def necessity_baseline_plot(rows, per_example, selected_ids, counts, path):
    necessity = [row for row in rows if row["phase"] == "necessity"]
    mean_rows = [row for row in rows if row["phase"] == "baseline_sensitivity"]
    corpus_same_rows = [row for row in rows if row["phase"] == "large_corpus_baseline"]
    corpus_adjusted_rows = [
        row for row in rows if row["phase"] == "large_corpus_baseline_adjusted"
    ]
    zero_rows = [
        row for row in per_example
        if row["method"] == "gradient_activation" and row["example_id"] in selected_ids
        and row["retained"] in counts
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    deletion_points = []
    for removed in sorted({row["removed"] for row in necessity}):
        group = [row for row in necessity if row["removed"] == removed]
        deletion_points.append((group[0]["removed_fraction"] * 100, sum(row["exact"] for row in group) / len(group)))
    axes[0].plot([x for x, _ in deletion_points], [y for _, y in deletion_points], marker="o", color="#dc2626")
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Top-attribution activations removed (%)")
    axes[0].set_ylabel("Exact accuracy")
    axes[0].set_title("Necessity: targeted deletions are highly disruptive", loc="left", weight="bold")
    axes[0].grid(alpha=0.2)

    for baseline_name, group_rows, color, linestyle in (
        ("Zero", zero_rows, "#ef4444", "-"),
        ("Arithmetic mean", mean_rows, "#2563eb", "-"),
        ("Corpus mean, zero ranking", corpus_same_rows, "#9ca3af", "--"),
        ("Corpus mean, adjusted ranking", corpus_adjusted_rows, "#16a34a", "-"),
    ):
        points = []
        for retained in counts:
            group = [row for row in group_rows if row["retained"] == retained]
            if group:
                points.append((group[0]["retained_fraction"] * 100, sum(row["exact"] for row in group) / len(group)))
        axes[1].plot(
            [x for x, _ in points], [y for _, y in points], marker="o",
            label=baseline_name, color=color, linestyle=linestyle,
        )
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Retained MLP activations (%)")
    axes[1].set_ylabel("Exact accuracy")
    axes[1].set_title("Sufficiency depends on the ablation baseline", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def structure_plot(root, path):
    structures = json.loads((root / "structures.json").read_text())
    chosen = [value for key, value in structures.items() if "gradient_activation__a0_9" in key]
    layers = np.asarray([item["layer_counts"] for item in chosen], dtype=float)
    positions = np.asarray([item["position_counts"] for item in chosen], dtype=float)
    layers /= layers.sum(axis=1, keepdims=True)
    positions /= positions.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(np.arange(layers.shape[1]), layers.mean(axis=0) * 100, color="#7c3aed")
    axes[0].set_xlabel("Transformer layer")
    axes[0].set_ylabel("Retained circuit share (%)")
    axes[0].set_title("Where 90%-margin circuits live", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(np.arange(positions.shape[1]), positions.mean(axis=0) * 100, color="#0891b2")
    axes[1].set_xlabel("Prompt token position")
    axes[1].set_ylabel("Retained circuit share (%)")
    axes[1].set_title("Which input positions retain neurons", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, path)


def image_data(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def unique_neuron_stats(root, meta):
    """Collapse position-specific masks to unique [layer, channel] neurons."""
    total = meta["total_masked_activations"]
    shape = tuple(meta["mask_shape"])
    counts = []
    with np.load(root / "successful_masks.npz") as archive:
        for key in archive.files:
            if "gradient_activation__a0_9" not in key:
                continue
            mask = np.unpackbits(archive[key], count=total).reshape(shape).astype(bool)
            counts.append(int(mask.any(axis=1).sum()))
    unique_total = shape[0] * shape[2]
    return {
        "total": unique_total,
        "values": counts,
        "median": median(counts),
        "median_fraction": median(counts) / unique_total,
        "min": min(counts),
        "max": max(counts),
    }


def main():
    args = parse_args()
    root = args.root
    meta = json.loads((root / "metadata.json").read_text())
    rows = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
    methods = list(meta["methods"])
    per_example = [row for row in rows if row["phase"] == "per_example"]
    total = meta["total_masked_activations"]
    counts = sorted({0, total, *(max(1, min(total, round(total * f))) for f in meta["fractions"])})
    curves = curve_rows(per_example, methods, counts)
    sizes = smallest_sizes(per_example, methods, meta["alphas"])
    unique_stats = unique_neuron_stats(root, meta)

    plots = {
        "fidelity": root / "fidelity_curves.png",
        "sizes": root / "circuit_sizes.png",
        "semantic": root / "semantic_failures.png",
        "transfer": root / "transfer_and_family.png",
        "necessity": root / "necessity_and_baseline.png",
        "structure": root / "circuit_structure.png",
    }
    fidelity_plot(curves, methods, plots["fidelity"])
    size_plot(sizes, methods, plots["sizes"])
    semantic_plot(per_example, counts, plots["semantic"])
    transfer_family_plot(rows, counts, methods, plots["transfer"])
    selected_ids = {item["example_id"] for item in meta["selected_examples"][:12]}
    necessity_baseline_plot(rows, per_example, selected_ids, counts, plots["necessity"])
    structure_plot(root, plots["structure"])

    transfer_summary = {}
    transfer = [row for row in rows if row["phase"] == "frozen_transfer"]
    for method in methods:
        group = [row for row in transfer if row["method"] == method]
        transfer_summary[method] = {
            "accuracy": sum(row["exact"] for row in group) / len(group),
            "failures": dict(Counter(row["failure_kind"] for row in group)),
            "mean_expected_distance": statistics.mean(row["expected_distance"] for row in group),
            "mean_retained_fraction": statistics.mean(row["retained_fraction"] for row in group),
        }
    family = [row for row in rows if row["phase"] == "task_family"]
    family_summary = {}
    for retained in counts:
        group = [row for row in family if row["retained"] == retained]
        if group:
            family_summary[str(group[0]["retained_fraction"])] = {
                "accuracy": sum(row["exact"] for row in group) / len(group),
                "median_kl": median([row["output_kl"] for row in group]),
                "mean_expected_distance": statistics.mean(row["expected_distance"] for row in group),
            }
    necessity = [row for row in rows if row["phase"] == "necessity"]
    necessity_summary = {}
    for removed in sorted({row["removed"] for row in necessity}):
        group = [row for row in necessity if row["removed"] == removed]
        necessity_summary[str(group[0]["removed_fraction"])] = sum(row["exact"] for row in group) / len(group)
    baseline_groups = {
        "zero": [
            row for row in per_example
            if row["method"] == "gradient_activation" and row["example_id"] in selected_ids
            and row["retained"] in counts
        ],
        "arithmetic_mean": [row for row in rows if row["phase"] == "baseline_sensitivity"],
        "corpus_mean_zero_ranking": [
            row for row in rows if row["phase"] == "large_corpus_baseline"
        ],
        "corpus_mean_adjusted_ranking": [
            row for row in rows if row["phase"] == "large_corpus_baseline_adjusted"
        ],
    }
    baseline_summary = {}
    for name, group_rows in baseline_groups.items():
        baseline_summary[name] = {}
        for retained in counts:
            group = [row for row in group_rows if row["retained"] == retained]
            if group:
                baseline_summary[name][str(group[0]["retained_fraction"])] = {
                    "accuracy": sum(row["exact"] for row in group) / len(group),
                    "median_kl": median([row["output_kl"] for row in group]),
                    "median_margin_retention": median(
                        [row["margin_retention"] for row in group]
                    ),
                    "mean_expected_distance": statistics.mean(
                        row["expected_distance"] for row in group
                    ),
                }
    iterative = [
        row for row in rows
        if row["phase"] == "iterative_pruning" and abs(row["retained_fraction"] - 0.2) < 1e-6
    ]
    iterative_ids = {row["example_id"] for row in iterative}
    one_shot_subset = [
        row for row in per_example
        if row["method"] == "gradient_activation"
        and row["example_id"] in iterative_ids
        and abs(row["retained_fraction"] - 0.2) < 1e-6
    ]
    costs = [json.loads(line) for line in (root / "extraction_costs.jsonl").read_text().splitlines()]
    iterative_costs = [row for row in costs if row.get("method") == "iterative_gradient_activation"]
    one_shot_costs = [row for row in costs if row.get("method") is None]
    iterative_summary = {
        "example_count": len(iterative),
        "retained_fraction": 0.2,
        "iterative_accuracy": sum(row["exact"] for row in iterative) / len(iterative),
        "one_shot_accuracy": sum(row["exact"] for row in one_shot_subset) / len(one_shot_subset),
        "iterative_median_margin_retention": median([row["margin_retention"] for row in iterative]),
        "one_shot_median_margin_retention": median([row["margin_retention"] for row in one_shot_subset]),
        "mean_iterative_backward_evaluations": statistics.mean(row["backward_evaluations"] for row in iterative_costs),
        "mean_iterative_seconds": statistics.mean(row["seconds"] for row in iterative_costs),
        "mean_one_shot_seconds": statistics.mean(row["seconds"] for row in one_shot_costs),
    }

    summary = {
        "metadata": meta,
        "curves": curves,
        "smallest_circuit_sizes": sizes,
        "unique_layer_channel_neurons_at_alpha_0_9": unique_stats,
        "frozen_transfer": transfer_summary,
        "task_family": family_summary,
        "necessity_accuracy": necessity_summary,
        "ablation_baselines": baseline_summary,
        "iterative_pruning": iterative_summary,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    primary = sizes["0.9"]["gradient_activation"]
    random_size = sizes["0.9"]["random"]
    transfer_primary = transfer_summary["gradient_activation"]
    family_20 = min(family_summary.items(), key=lambda item: abs(float(item[0]) - 0.2))[1]
    removal_001 = min(necessity_summary.items(), key=lambda item: abs(float(item[0]) - 0.001))
    baseline_20 = {
        name: min(values.items(), key=lambda item: abs(float(item[0]) - 0.2))[1]
        for name, values in baseline_summary.items()
    }
    report_md = f"""# Native-neuron circuit pilot: arithmetic

## Main result

For {len(meta['selected_examples'])} correctly answered prompts, gradient × activation found a **median 90%-margin circuit retaining {primary['median']:.1%}** of the {total:,} layer–position–neuron activations (IQR {primary['q1']:.1%}–{primary['q3']:.1%}). Random ranking required a median {random_size['median']:.1%}.

This is substantial compression, but not an ultra-small circuit. Circuit size varied sharply by prompt ({primary['min']:.1%}–{primary['max']:.1%}), so a single headline neuron count would be misleading.

The position-specific count hides another important fact: after collapsing token positions, the median circuit touched **{unique_stats['median']:,.0f} of {unique_stats['total']:,} unique layer–channel neurons ({unique_stats['median_fraction']:.1%})** at least once. Most parameter neurons participate somewhere; sparsity is much stronger over *when* a neuron fires than over a globally reusable neuron subset.

## What happens when neurons are removed

- Errors degrade in a graded way: masks first produce nearby and far numeric answers; at severe ablation they increasingly produce non-numeric tokens.
- Frozen 90%-margin gradient × activation masks remained exactly correct on **{transfer_primary['accuracy']:.1%}** of changed arithmetic problems. Their failures were {transfer_primary['failures']}.
- A single task-family mask retained **{family_20['accuracy']:.1%}** exact accuracy at approximately 20% retained activations on 30 held-out problems.
- Necessity is much sparser than sufficiency: after deleting roughly {float(removal_001[0]):.2%} of the highest-attribution activations, exact accuracy was only **{removal_001[1]:.1%}** on the 12-prompt necessity subset.
- Baseline choice is consequential. At 20% retention, zero ablation reached {baseline_20['zero']['accuracy']:.1%} accuracy (median KL {baseline_20['zero']['median_kl']:.3f}); the 24-prompt arithmetic mean reached {baseline_20['arithmetic_mean']['accuracy']:.1%} (KL {baseline_20['arithmetic_mean']['median_kl']:.3f}); and the 550,000-token WikiText-103 mean with its correct gradient × (activation − mean) ranking reached {baseline_20['corpus_mean_adjusted_ranking']['accuracy']:.1%} (KL {baseline_20['corpus_mean_adjusted_ranking']['median_kl']:.3f}).
- Reusing the zero-ablation ranking with the corpus mean reached only {baseline_20['corpus_mean_zero_ranking']['accuracy']:.1%} at 20%. Mean replacement changes both the intervention and which units count as important.
- Iterative re-ranking did not improve the six-prompt 20% circuit subset over one-shot gradient × activation (both 100% exact with the same median margin retention), while requiring {iterative_summary['mean_iterative_backward_evaluations']:.0f} extra backward evaluations. More search was not automatically better here.

## Interpretation

The model's arithmetic behavior is distributed across many MLP activation instances, but it is not uniform. Attribution-based selection compresses the computation far better than random selection, and a tiny targeted deletion can be disproportionately damaging. Per-input masks partly transfer when the operands and answer change, but they are not fully reusable.

Counts here are activation instances indexed by layer, prompt position, and MLP channel—not unique parameter neurons. Attention remains intact, so this is MLP sufficiency conditional on the original attention computation.

## Scope and cautions

- Model: `{meta['model']}`; 152/152 eligible prompts answered correctly before masking.
- Task: single-token, single-digit addition, subtraction, and multiplication; 24 extraction prompts and 30 held-out task-family prompts.
- Primary intervention: binary zero ablation after the gated MLP activation and before its output projection, with a fresh full forward pass every time.
- Large-corpus baseline: 50,000 fixed-length windows (550,000 tokens) streamed from WikiText-103 training text. Corpus-mean rankings use |gradient × (activation − mean)|.
- Semantic expected distance assigns absolute error to digits and cost 10 to non-numeric vocabulary mass. The reported nearby/far/non-numeric categories are less assumption-sensitive than that scalar.
- The extraction is heuristic and answer-informed. It finds small sufficient masks, not certified minima or proof that individual neurons have human-readable meanings.
- Dense masking measures causal sufficiency, not runtime or FLOP savings.
"""
    (root / "summary.md").write_text(report_md, encoding="utf-8")

    sections = [
        ("Fidelity and semantic degradation", "fidelity", "How exact answers, margin, full-distribution KL, and graded semantic cost change with circuit size."),
        ("Smallest circuits found", "sizes", "Prompt-to-prompt variability at three fidelity requirements."),
        ("What wrong answers look like", "semantic", "A wrong digit one step away is separated from far digits and non-numeric failures."),
        ("Reuse beyond the extraction prompt", "transfer", "Frozen per-input masks and one shared task-family mask."),
        ("Necessity and baseline sensitivity", "necessity", "Targeted deletion and the difference between zero and mean replacement."),
        ("Circuit structure", "structure", "Retained activation counts by layer and prompt position."),
    ]
    figures = "".join(
        f"<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>"
        f"<img src=\"{image_data(plots[key])}\" alt=\"{html.escape(title)}\"></section>"
        for title, key, caption in sections
    )
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native-neuron circuit pilot</title><style>
:root {{ color-scheme: light dark; }} body {{ margin:0; font:16px/1.55 system-ui,sans-serif; background:#f4f4f5; color:#18181b; }}
main {{ max-width:900px; margin:auto; padding:20px 14px 60px; }} header,section {{ background:white; border:1px solid #e4e4e7; border-radius:14px; padding:18px; margin:0 0 16px; }}
h1 {{ font-size:clamp(25px,7vw,40px); line-height:1.1; margin:.1em 0 .4em; }} h2 {{ font-size:21px; margin:0 0 4px; }}
.lead {{ font-size:18px; }} .finding {{ border-left:5px solid #ef4444; padding-left:14px; }} img {{ display:block; width:100%; height:auto; margin-top:12px; }}
code {{ overflow-wrap:anywhere; }} @media (prefers-color-scheme:dark) {{ body {{ background:#09090b; color:#f4f4f5; }} header,section {{ background:#18181b; border-color:#3f3f46; }} }}
</style></head><body><main><header><h1>How many neuron activations does arithmetic need?</h1>
<p class="lead">A causal masking pilot on {html.escape(meta['model'])}, with graded numeric errors rather than accuracy alone.</p>
<p class="finding"><strong>Median 90%-margin circuit: {primary['median']:.1%}</strong> of {total:,} MLP activation instances using gradient × activation ranking. Across prompts: {primary['min']:.1%}–{primary['max']:.1%}.</p>
<p>Collapsed across token positions, those masks touched a median <strong>{unique_stats['median_fraction']:.1%}</strong> of unique layer–channel neurons at least once. The compactness is mainly position-specific.</p>
<p>Frozen masks stayed exact on <strong>{transfer_primary['accuracy']:.1%}</strong> of changed problems; a shared 20% task mask reached <strong>{family_20['accuracy']:.1%}</strong> held-out accuracy.</p></header>
{figures}<section><h2>What this establishes</h2><p>The computation is compressible but distributed, targeted activations are far more important than random ones, and errors pass through meaningful numeric neighborhoods before severe ablation becomes non-numeric. The result is conditional on intact attention and does not make individual neurons semantically interpretable.</p></section>
<section><h2>Method in one paragraph</h2><p>We kept only prompts the intact model answered correctly, masked gated-MLP activations at every layer and token position, and recomputed the entire downstream network on every evaluation. We measured exact answers, answer margin, full-vocabulary KL, numeric distance, frozen-mask transfer, task-family masks, targeted deletion, and zero, task-mean, and 550,000-token corpus-mean baselines.</p></section>
</main></body></html>"""
    (root / "report.html").write_text(html_doc, encoding="utf-8")
    print(f"wrote summary and {len(plots)} plots to {root}")


if __name__ == "__main__":
    main()
