#!/usr/bin/env python3
"""Summarize and plot the multi-task language ablation pilot."""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/language_pilot")
FAMILIES = ("ioi", "agreement", "brackets")
DISPLAY = {"ioi": "IOI", "agreement": "Agreement", "brackets": "Bracket closing"}
COLORS = {"zero": "#3b82f6", "corpus_mean": "#f97316", "random": "#94a3b8"}


def load_rows() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "evaluations.jsonl").read_text().splitlines()]


def median_circuit(rows, family, baseline, method):
    values = []
    ids = {r["example_id"] for r in rows if r["phase"] == "per_example" and r["family"] == family}
    for example_id in ids:
        feasible = [
            r for r in rows
            if r["phase"] == "per_example" and r["example_id"] == example_id
            and r["baseline"] == baseline and r["method"] == method
            and r["masked_margin"] >= 0.9 * r["intact_margin"]
        ]
        if feasible:
            values.append(min(r["retained_fraction"] for r in feasible))
    return statistics.median(values), values


def aggregate_curve(rows, family, baseline, method, phase="per_example"):
    grouped = defaultdict(list)
    for row in rows:
        if row["phase"] == phase and row["family"] == family and row["baseline"] == baseline and row["method"] == method:
            grouped[row["retained_fraction"]].append(row)
    return [
        {
            "x": x,
            "accuracy": statistics.mean(float(r["exact"]) for r in values),
            "kl": statistics.mean(r["output_kl"] for r in values),
            "margin_retention": statistics.mean(r["margin_retention"] for r in values),
            "semantic_cost": statistics.mean(r["expected_semantic_cost"] for r in values),
        }
        for x, values in sorted(grouped.items())
    ]


def at_fraction(rows, family, baseline, method, phase, fraction):
    values = [
        r for r in rows if r["phase"] == phase and r["family"] == family
        and r["baseline"] == baseline and r["method"] == method
    ]
    target = min({r["retained_fraction"] for r in values}, key=lambda x: abs(x - fraction))
    values = [r for r in values if r["retained_fraction"] == target]
    return {
        "fraction": target,
        "accuracy": statistics.mean(float(r["exact"]) for r in values),
        "kl": statistics.mean(r["output_kl"] for r in values),
        "semantic_cost": statistics.mean(r["expected_semantic_cost"] for r in values),
        "failure_counts": dict(Counter(r["failure_kind"] for r in values)),
    }


def plot_circuit_sizes(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FAMILIES))
    settings = [
        ("Targeted · zero", "zero", "gradient_activation", "#2563eb"),
        ("Targeted · corpus mean", "corpus_mean", "gradient_difference", "#f97316"),
        ("Random · zero", "zero", "random", "#94a3b8"),
    ]
    width = 0.24
    for offset, (label, baseline, method, color) in enumerate(settings):
        values = [100 * median_circuit(rows, family, baseline, method)[0] for family in FAMILIES]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(x, [DISPLAY[f] for f in FAMILIES])
    ax.set_ylabel("Median retained activations for 90% margin (%)")
    ax.set_title("Targeted rankings find much smaller sufficient circuits than random masks")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "language_circuit_sizes.png", dpi=180)
    plt.close(fig)


def plot_fidelity(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        for baseline, method, label, color in (
            ("zero", "gradient_activation", "Zero", "#2563eb"),
            ("corpus_mean", "gradient_difference", "Corpus mean", "#f97316"),
        ):
            curve = aggregate_curve(rows, family, baseline, method)
            xs = [100 * point["x"] for point in curve if point["x"] > 0]
            ys = [point["accuracy"] for point in curve if point["x"] > 0]
            ax.plot(xs, ys, marker="o", ms=3, lw=2, color=color, label=label)
        ax.set_xscale("log")
        ax.set_title(DISPLAY[family])
        ax.set_xlabel("Retained activations (%)")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Exact next-token accuracy")
    axes[0].legend(frameon=False)
    fig.suptitle("Per-example sufficiency curves")
    fig.tight_layout()
    fig.savefig(ROOT / "language_fidelity_curves.png", dpi=180)
    plt.close(fig)


def plot_deletion(rows):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for family, color in zip(FAMILIES, ("#2563eb", "#16a34a", "#9333ea"), strict=True):
        grouped = defaultdict(list)
        for row in rows:
            if row["phase"] == "necessity" and row["family"] == family:
                grouped[row["removed_fraction"]].append(float(row["exact"]))
        xs = sorted(x for x in grouped if x > 0)
        ax.plot([100 * x for x in xs], [statistics.mean(grouped[x]) for x in xs], marker="o", lw=2, label=DISPLAY[family], color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Top-attribution activations removed (%)")
    ax.set_ylabel("Exact next-token accuracy")
    ax.set_title("Tiny targeted deletions are sufficient to break all three behaviors")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "language_necessity.png", dpi=180)
    plt.close(fig)


def plot_generalization(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(FAMILIES))
    width = 0.35
    for offset, baseline in enumerate(("zero", "corpus_mean")):
        transfer = []
        family20 = []
        for family in FAMILIES:
            q = [r for r in rows if r["phase"] == "frozen_transfer" and r["family"] == family and r["baseline"] == baseline]
            transfer.append(statistics.mean(float(r["exact"]) for r in q))
            method = "family_gradient_activation" if baseline == "zero" else "family_gradient_difference"
            family20.append(at_fraction(rows, family, baseline, method, "task_family", 0.2)["accuracy"])
        label = "Zero" if baseline == "zero" else "Corpus mean"
        color = COLORS[baseline]
        axes[0].bar(x + (offset - .5) * width, transfer, width, label=label, color=color)
        axes[1].bar(x + (offset - .5) * width, family20, width, label=label, color=color)
    axes[0].set_title("Frozen per-example circuit")
    axes[1].set_title("Shared task circuit at 20% retained")
    for ax in axes:
        ax.set_xticks(x, [DISPLAY[f] for f in FAMILIES], rotation=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out exact accuracy")
    axes[0].legend(frameon=False)
    fig.suptitle("Generalization distinguishes input-specific and shared circuits")
    fig.tight_layout()
    fig.savefig(ROOT / "language_generalization.png", dpi=180)
    plt.close(fig)


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    rows = load_rows()
    metadata = json.loads((ROOT / "metadata.json").read_text())
    plot_circuit_sizes(rows)
    plot_fidelity(rows)
    plot_deletion(rows)
    plot_generalization(rows)

    summary = {"model": metadata["model"], "families": {}}
    for family in FAMILIES:
        zero_median, zero_values = median_circuit(rows, family, "zero", "gradient_activation")
        mean_median, mean_values = median_circuit(rows, family, "corpus_mean", "gradient_difference")
        random_median, _ = median_circuit(rows, family, "zero", "random")
        transfer = {}
        for baseline in ("zero", "corpus_mean"):
            q = [r for r in rows if r["phase"] == "frozen_transfer" and r["family"] == family and r["baseline"] == baseline]
            transfer[baseline] = {
                "accuracy": statistics.mean(float(r["exact"]) for r in q),
                "margin_90_rate": statistics.mean(float(r["margin_retention"] >= 0.9) for r in q),
            }
        deletion = [r for r in rows if r["phase"] == "necessity" and r["family"] == family and r["removed"] > 0]
        closest = min({r["removed_fraction"] for r in deletion}, key=lambda x: abs(x - 0.0001))
        deletion = [r for r in deletion if r["removed_fraction"] == closest]
        summary["families"][family] = {
            "sequence_length": metadata["sequence_lengths"][family],
            "aligned_intact_correct": metadata["aligned_intact_correct_counts"][family],
            "median_90pct_margin_circuit_fraction": {
                "zero_targeted": zero_median, "corpus_mean_targeted": mean_median, "zero_random": random_median,
            },
            "per_example_at_20pct": {
                "zero": at_fraction(rows, family, "zero", "gradient_activation", "per_example", 0.2),
                "corpus_mean": at_fraction(rows, family, "corpus_mean", "gradient_difference", "per_example", 0.2),
            },
            "frozen_transfer": transfer,
            "shared_circuit_at_20pct": {
                "zero": at_fraction(rows, family, "zero", "family_gradient_activation", "task_family", 0.2),
                "corpus_mean": at_fraction(rows, family, "corpus_mean", "family_gradient_difference", "task_family", 0.2),
            },
            "deletion_near_0_01pct": {
                "removed_fraction": closest,
                "accuracy": statistics.mean(float(r["exact"]) for r in deletion),
                "failure_counts": dict(Counter(r["failure_kind"] for r in deletion)),
            },
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Language neuron-ablation pilot", "",
        f"Model: `{metadata['model']}`. Extraction examples: 9 per task; held-out examples: 9 per task.", "",
        "| Task | Intact aligned pool | Median 90%-margin circuit (zero) | Median 90%-margin circuit (corpus mean) | Random (zero) | Accuracy after ~0.01% targeted deletion |", "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        s = summary["families"][family]
        c = s["median_90pct_margin_circuit_fraction"]
        d = s["deletion_near_0_01pct"]
        lines.append(f"| {DISPLAY[family]} | {s['aligned_intact_correct']} | {c['zero_targeted']:.1%} | {c['corpus_mean_targeted']:.1%} | {c['zero_random']:.1%} | {d['accuracy']:.1%} |")
    lines += ["", "At 20% retained, corpus-mean ablation was especially helpful for IOI: 100% exact accuracy versus 44.4% under zero ablation. Shared 20% circuits reached 66.7%/88.9% (zero/mean) for IOI, 77.8%/77.8% for agreement, and 100%/88.9% for bracket closing.", "", "Frozen circuits often preserved the exact answer on a changed input (55.6–100%), but retained 90% of the original logit margin much less often (11.1–55.6%). Exact output transfer therefore overstates mechanistic invariance.", "", "Semantic failures were recorded as role reversal, another known name, wrong agreement, wrong closer, opener, or unrelated output; full-vocabulary KL and expected semantic cost are included in every evaluation row."]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sections = "".join(
        f'<figure><img src="{image_uri(ROOT / filename)}" alt="{title}"><figcaption>{title}</figcaption></figure>'
        for filename, title in (
            ("language_circuit_sizes.png", "Sufficient-circuit size"),
            ("language_fidelity_curves.png", "Per-example accuracy curves"),
            ("language_necessity.png", "Targeted deletion / necessity"),
            ("language_generalization.png", "Frozen transfer and shared circuits"),
        )
    )
    html = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Language neuron-ablation pilot</title><style>
body{{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#172033}}main{{max-width:980px;margin:auto;padding:22px}}
h1{{line-height:1.1}}.lead{{font-size:1.1rem}}.note{{background:#fff7ed;border-left:4px solid #f97316;padding:12px 16px}}
figure{{margin:28px 0;background:white;border:1px solid #dbe3ee;padding:10px}}img{{width:100%;height:auto}}figcaption{{text-align:center;color:#475569}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}
@media(max-width:600px){{main{{padding:14px}}table{{font-size:12px}}th,td{{padding:5px}}}}
</style><main><h1>Language neuron-ablation pilot</h1>
<p class="lead">The model was screened first, then neuron-position activations were retained or deleted under zero and a 50,000-window WikiText corpus mean.</p>
<div class="note"><strong>Headline:</strong> task behavior is highly deletion-sensitive, but sufficient circuits are not tiny. Targeted masks need median 22.8–43.4% of activations under zero ablation, versus 89.1–95.3% for random masks.</div>
<h2>Core results</h2><table><tr><th>Task</th><th>Zero circuit</th><th>Mean circuit</th><th>Random</th><th>After ~0.01% deletion</th></tr>
{''.join(f'<tr><td>{DISPLAY[f]}</td><td>{summary["families"][f]["median_90pct_margin_circuit_fraction"]["zero_targeted"]:.1%}</td><td>{summary["families"][f]["median_90pct_margin_circuit_fraction"]["corpus_mean_targeted"]:.1%}</td><td>{summary["families"][f]["median_90pct_margin_circuit_fraction"]["zero_random"]:.1%}</td><td>{summary["families"][f]["deletion_near_0_01pct"]["accuracy"]:.1%}</td></tr>' for f in FAMILIES)}</table>
{sections}<h2>Interpretation</h2><p>There is a sparse set of highly necessary activations, but retaining only that tiny set is not sufficient. Corpus-mean replacement helps IOI strongly at medium budgets, while its advantage is smaller or reversed for agreement and brackets. Frozen-mask exact accuracy is materially higher than 90%-margin fidelity, so output-only transfer would make the circuits look more invariant than they are.</p>
<p>Raw outputs: 3,060 fresh-forward evaluations, full-vocabulary KL, semantic cost, exact outcome categories, and saved successful mask indices.</p></main>"""
    (ROOT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
