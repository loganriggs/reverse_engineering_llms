#!/usr/bin/env python3
"""Summarize attention counterfactual-patching results."""

from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


ROOT = Path("artifacts/attention_counterfactuals")
MLP_ROOT = Path("artifacts/counterfactual_patching")
FAMILIES = ("ioi", "agreement", "brackets")
LABELS = {"ioi": "IOI", "agreement": "Agreement", "brackets": "Brackets"}
COMPONENTS = ("query", "key", "value", "pattern", "message", "head_output")
COMPONENT_LABELS = {
    "query": "Query", "key": "Key", "value": "Value", "pattern": "Attention row",
    "message": "Message", "head_output": "Head output",
}


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def crossover(rows, family, component, method="counterfactual_gradient"):
    values = {}
    pair_ids = {row["pair_id"] for row in rows if row["family"] == family}
    for pair_id in pair_ids:
        success = [
            row for row in rows if row["pair_id"] == pair_id
            and row["component"] == component and row["method"] == method
            and row["target_top1"]
        ]
        if success:
            values[pair_id] = min(row["patched"] for row in success)
    return values


def mlp_crossover(rows, family):
    values = {}
    for pair_id in {row["pair_id"] for row in rows if row["family"] == family}:
        success = [
            row for row in rows if row["pair_id"] == pair_id
            and row["resolution"] == "instance"
            and row["method"] == "counterfactual_gradient" and row["target_top1"]
        ]
        values[pair_id] = min(row["patched"] for row in success)
    return values


def plot_routing_content(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(COMPONENTS))
    colors = ("#2563eb", "#16a34a", "#9333ea")
    width = .24
    for family_index, (family, color) in enumerate(zip(FAMILIES, colors, strict=True)):
        counts = [len(crossover(rows, family, component)) for component in COMPONENTS]
        medians = []
        for component in COMPONENTS:
            values = list(crossover(rows, family, component).values())
            medians.append(statistics.median(values) if values else np.nan)
        axes[0].bar(x + (family_index - 1) * width, counts, width, label=LABELS[family], color=color)
        axes[1].scatter(x + (family_index - 1) * .12, medians, s=65, label=LABELS[family], color=color)
    for ax in axes:
        ax.set_xticks(x, [COMPONENT_LABELS[c] for c in COMPONENTS], rotation=18, ha="right")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pairs flipped out of 12")
    axes[0].set_ylim(0, 12.8)
    axes[0].set_title("Can this intervention produce the answer?")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median units at strict top-1 crossover")
    axes[1].set_title("How many targeted units are needed?")
    axes[0].legend(frameon=False, ncols=3)
    fig.suptitle("Routing patches and content patches separate by task")
    fig.tight_layout()
    fig.savefig(ROOT / "attention_routing_vs_content.png", dpi=180)
    plt.close(fig)


def plot_controls(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    selected_components = ("value", "message", "head_output")
    x = np.arange(len(selected_components))
    for ax, family in zip(axes, FAMILIES, strict=True):
        targeted, random = [], []
        for component in selected_components:
            target_values = list(crossover(rows, family, component).values())
            random_values = list(crossover(rows, family, component, "random").values())
            targeted.append(statistics.median(target_values))
            random.append(statistics.median(random_values))
        ax.bar(x - .18, targeted, .36, label="Gradient × change", color="#2563eb")
        ax.bar(x + .18, random, .36, label="Random", color="#94a3b8")
        ax.set_yscale("log")
        ax.set_xticks(x, [COMPONENT_LABELS[c] for c in selected_components], rotation=15)
        ax.set_title(LABELS[family])
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median units at strict top-1 crossover")
    axes[0].legend(frameon=False)
    fig.suptitle("Sparse attention control is not reproduced by random patches")
    fig.tight_layout()
    fig.savefig(ROOT / "attention_random_controls.png", dpi=180)
    plt.close(fig)


def plot_mlp_comparison(rows, mlp_rows):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(FAMILIES))
    series = (
        ("MLP scalar instance", [statistics.median(mlp_crossover(mlp_rows, f).values()) for f in FAMILIES], "#64748b"),
        ("Attention message", [statistics.median(crossover(rows, f, "message").values()) for f in FAMILIES], "#f97316"),
        ("Attention head output", [statistics.median(crossover(rows, f, "head_output").values()) for f in FAMILIES], "#2563eb"),
    )
    width = .24
    for index, (label, values, color) in enumerate(series):
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%g", padding=2)
    ax.set_xticks(x, [LABELS[f] for f in FAMILIES])
    ax.set_ylabel("Median targeted units for strict top-1 flip")
    ax.set_title("Attention vectors are coarser units than scalar MLP activations")
    ax.legend(frameon=False, ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "attention_vs_mlp.png", dpi=180)
    plt.close(fig)


def decode_examples(metadata):
    tokenizer = AutoTokenizer.from_pretrained(metadata["model"])
    pair_meta = json.loads((MLP_ROOT / "metadata.json").read_text())
    pairs = {pair["pair_id"]: pair for pair in pair_meta["selected_pairs"]}
    arrays = np.load(ROOT / "crossover_indices.npz")
    chosen = {
        "ioi": "ioi-john-mary-forward",
        "agreement": "agreement-book-picture-sg-to-pl",
        "brackets": "brackets-40-to-91-3",
    }
    output = {}
    for family, pair_id in chosen.items():
        pair = pairs[pair_id]
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(pair["base"]["prompt"], add_special_tokens=False)
        )
        length = len(tokens)
        messages = []
        for raw in arrays[f"{pair_id}__message__top1"]:
            index = int(raw)
            layer = index // (14 * length * length)
            remainder = index % (14 * length * length)
            head = remainder // (length * length)
            remainder %= length * length
            destination, source = divmod(remainder, length)
            messages.append({
                "layer": layer, "head": head,
                "destination": destination, "destination_token": tokens[destination],
                "source": source, "source_token": tokens[source],
            })
        output[family] = {
            "pair_id": pair_id, "base_prompt": pair["base"]["prompt"],
            "base_answer": pair["base"]["answer_text"],
            "counterfactual_prompt": pair["source"]["prompt"],
            "counterfactual_answer": pair["source"]["answer_text"],
            "messages": messages,
        }
    return output


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    rows = load_rows(ROOT / "evaluations.jsonl")
    mlp_rows = load_rows(MLP_ROOT / "evaluations.jsonl")
    metadata = json.loads((ROOT / "metadata.json").read_text())
    plot_routing_content(rows)
    plot_controls(rows)
    plot_mlp_comparison(rows, mlp_rows)
    examples = decode_examples(metadata)
    summary = {"model": metadata["model"], "architecture": metadata, "families": {}, "examples": examples}
    for family in FAMILIES:
        summary["families"][family] = {}
        for component in (*COMPONENTS, "query_key_joint"):
            values = list(crossover(rows, family, component).values())
            random_values = [] if component == "query_key_joint" else list(crossover(rows, family, component, "random").values())
            summary["families"][family][component] = {
                "successes": len(values),
                "median": statistics.median(values) if values else None,
                "range": [min(values), max(values)] if values else None,
                "random_successes": len(random_values),
                "random_median": statistics.median(random_values) if random_values else None,
            }
        summary["families"][family]["mlp_instance_median"] = statistics.median(
            mlp_crossover(mlp_rows, family).values()
        )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Attention counterfactual patching", "",
        "Strict crossover means the donor answer became the full-vocabulary top-1 token in a fresh forward pass.", "",
        "| Task | Query | Key | Value | Pattern | Message | Head output | MLP scalar |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        s = summary["families"][family]
        cell = lambda component: f"{s[component]['median']:g} ({s[component]['successes']}/12)" if s[component]["median"] is not None else f"— (0/12)"
        lines.append(f"| {LABELS[family]} | {cell('query')} | {cell('key')} | {cell('value')} | {cell('pattern')} | {cell('message')} | {cell('head_output')} | {s['mlp_instance_median']:g} |")
    lines += ["", "Entries are median targeted units followed by successful pairs. One value or head-output unit is a 64-dimensional head vector; one message is a 64-dimensional source-to-destination contribution. An MLP unit is one scalar activation instance, so the counts are not dimension-for-dimension equivalents.", "", "Agreement is the cleanest separation: query, key, pattern, and Q+K patches never flipped any pair, while exactly one value vector, one message, or one head-output vector flipped every pair. IOI uses both routing and content. Bracket behavior is predominantly content-controlled in this intervention family.", "", "## IOI example", "", "For John/Mary, two messages flip Mary to John: layer 20 head 9 from the repeated-subject token `John` to the final `to`, and layer 23 head 9 from the initial `Mary` token to the final `to`."]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figures = "".join(
        f'<figure><img src="{image_uri(ROOT / filename)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        for filename, caption in (
            ("attention_routing_vs_content.png", "Routing and content interventions by task"),
            ("attention_random_controls.png", "Targeted rankings versus random patches"),
            ("attention_vs_mlp.png", "Attention-unit and MLP-unit crossover counts"),
        )
    )
    table = "".join(
        f"<tr><td>{LABELS[f]}</td>" + "".join(
            f"<td>{summary['families'][f][c]['median']:g} ({summary['families'][f][c]['successes']}/12)</td>" if summary['families'][f][c]['median'] is not None else "<td>— (0/12)</td>"
            for c in ("query", "key", "value", "pattern", "message", "head_output")
        ) + "</tr>" for f in FAMILIES
    )
    html = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attention counterfactual patching</title><style>body{{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#172033}}main{{max-width:980px;margin:auto;padding:22px}}.lead{{font-size:1.1rem}}.note{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px}}figure{{margin:28px 0;background:white;border:1px solid #dbe3ee;padding:10px}}img{{width:100%;height:auto}}figcaption{{text-align:center;color:#475569}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:720px;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:600px){{main{{padding:14px}}table{{font-size:12px}}}}</style><main><h1>Attention counterfactual patching</h1><p class="lead">Donor queries, keys, values, normalized rows, source→destination messages, and whole head outputs are patched separately on the same 36 language counterfactuals.</p><div class="note"><strong>Headline:</strong> IOI uses both routing and payload. Agreement and bracket closing are controlled much more reliably through values/messages/head outputs than through queries, keys, or attention rows.</div><h2>Median units at strict top-1 crossover</h2><div class="scroll"><table><tr><th>Task</th><th>Query</th><th>Key</th><th>Value</th><th>Pattern</th><th>Message</th><th>Head output</th></tr>{table}</table></div>{figures}<h2>Concrete IOI route</h2><p>For “John and Mary … John gave … to,” two donor messages flip the answer from Mary to John: layer 20/head 9 carries the repeated-subject position into the final prediction position, and layer 23/head 9 carries the initial Mary position into that same destination.</p><p>A head vector or message is 64-dimensional, whereas an MLP activation instance is scalar. Their raw counts answer different granularity questions and should not be treated as equal-sized parameterizations.</p></main>"""
    (ROOT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
