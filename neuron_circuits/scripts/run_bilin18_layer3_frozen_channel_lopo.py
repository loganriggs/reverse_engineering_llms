#!/usr/bin/env python3
"""Red-team frozen Layer-3 family channels with leave-one-pair-out discovery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    evaluate_routes,
    recover_configs,
    top_indices,
)
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_frozen_channel_lopo.json")
OUTPUT = TENSOR_ROOT / "layer3_frozen_channel_lopo.json"
REPORT = TENSOR_ROOT / "LAYER3_FROZEN_CHANNEL_LOPO.md"
FIGURE = TENSOR_ROOT / "layer3_frozen_channel_lopo.png"


def metric_summary(records, route):
    return {
        metric: median([record["routes"][route][metric] for record in records])
        for metric in (
            "local_fidelity", "downstream_fidelity", "downstream_cosine",
            "target_projection", "energy_ratio",
        )
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    frozen_config = json.loads(Path(config["source_config"]).read_text())
    native_config = json.loads(Path(frozen_config["source_config"]).read_text())
    native_saved = json.loads(Path(frozen_config["source_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    rank, layer = config["rank"], native_config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    specs = fixed["evaluation"]

    def reconstruct(record):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(
            rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
        return base[layer], counterfactual[layer]

    discovery_records = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    discovery = {}
    family_by_pair = {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        score = channel_state(model, layer, base, counterfactual, final_position)["scores"].cpu()
        discovery[record["pair"]] = score
        family_by_pair[record["pair"]] = record["family"]
        print("discovery", record["family"], record["pair"], flush=True)

    family_names = list(parent["families"])
    inclusive_family = {
        family: top_indices(torch.stack([
            score.abs() for pair, score in discovery.items() if family_by_pair[pair] == family
        ]).mean(0), rank).cuda()
        for family in family_names
    }
    generator = torch.Generator().manual_seed(frozen_config["seed"])
    random_set = torch.randperm(next(iter(discovery.values())).numel(), generator=generator)[:rank].cuda()
    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    for record in query_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        family = record["family"]
        family_lopo_scores = [
            score.abs() for candidate_pair, score in discovery.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != record["pair"]
        ]
        global_lopo_scores = [
            score.abs() for candidate_pair, score in discovery.items()
            if candidate_pair != record["pair"]
        ]
        wrong_names = [name for name in family_names if name != family]
        routes = {
            "oracle": top_indices(state["scores"], rank),
            "pair": top_indices(discovery[record["pair"]], rank).cuda(),
            "family_inclusive": inclusive_family[family],
            "family_lopo": top_indices(torch.stack(family_lopo_scores).mean(0), rank).cuda(),
            "global_lopo": top_indices(torch.stack(global_lopo_scores).mean(0), rank).cuda(),
            "random": random_set,
        }
        for index, wrong in enumerate(wrong_names):
            routes[f"wrong_{index}"] = inclusive_family[wrong]
        result = evaluate_routes(model, layer, base, counterfactual, state, routes, final_position)
        wrong_results = [result[f"wrong_{index}"] for index in range(len(wrong_names))]
        result["wrong_family_median"] = {
            metric: float(np.median([row[metric] for row in wrong_results]))
            for metric in wrong_results[0]
        }
        best = max(wrong_results, key=lambda row: row["downstream_fidelity"])
        result["wrong_family_best"] = dict(best)
        oracle_set = set(routes["oracle"].tolist())
        support = {
            name: len(set(routes[name].tolist()) & oracle_set)
            / len(set(routes[name].tolist()) | oracle_set)
            for name in ("pair", "family_inclusive", "family_lopo", "global_lopo", "random")
        }
        evaluated.append({
            "corpus": record["corpus"],
            "family": family,
            "pair": record["pair"],
            "routes": result,
            "support_jaccard": support,
        })
        print("evaluation", record["corpus"], family, record["pair"], flush=True)

    displayed = [
        "oracle", "pair", "family_inclusive", "family_lopo", "global_lopo",
        "wrong_family_median", "wrong_family_best", "random",
    ]
    summaries = {}
    for corpus in fixed["evaluation"]:
        records = [record for record in evaluated if record["corpus"] == corpus]
        summaries[corpus] = {name: metric_summary(records, name) for name in displayed}
        summaries[corpus]["support_jaccard"] = {
            name: median([record["support_jaccard"][name] for record in records])
            for name in ("pair", "family_inclusive", "family_lopo", "global_lopo", "random")
        }
        summaries[corpus]["by_family"] = {
            family: {
                name: metric_summary(
                    [record for record in records if record["family"] == family], name,
                )["downstream_fidelity"]
                for name in (
                    "pair", "family_inclusive", "family_lopo", "global_lopo",
                    "wrong_family_median", "wrong_family_best", "random",
                )
            }
            for family in family_names
        }

    fidelity = lambda corpus, route: summaries[corpus][route]["downstream_fidelity"]
    hypotheses = {
        "E1_lopo_family_transfers": {
            "supported": all(fidelity(c, "family_lopo") >= 0.15 for c in summaries),
        },
        "E2_lopo_family_beats_random": {
            "supported": all(fidelity(c, "family_lopo") - fidelity(c, "random") >= 0.05 for c in summaries),
            "gaps": {c: fidelity(c, "family_lopo") - fidelity(c, "random") for c in summaries},
        },
        "E3_lopo_family_beats_median_wrong_family": {
            "supported": all(fidelity(c, "family_lopo") - fidelity(c, "wrong_family_median") >= 0.05 for c in summaries),
            "gaps": {c: fidelity(c, "family_lopo") - fidelity(c, "wrong_family_median") for c in summaries},
        },
        "E4_lopo_family_beats_best_wrong_family": {
            "supported": all(fidelity(c, "family_lopo") > fidelity(c, "wrong_family_best") for c in summaries),
            "gaps": {c: fidelity(c, "family_lopo") - fidelity(c, "wrong_family_best") for c in summaries},
        },
        "E5_lopo_family_beats_lopo_global": {
            "supported": all(fidelity(c, "family_lopo") - fidelity(c, "global_lopo") >= 0.03 for c in summaries),
            "gaps": {c: fidelity(c, "family_lopo") - fidelity(c, "global_lopo") for c in summaries},
        },
        "E6_lopo_family_transfers_ood": {
            "supported": fidelity("tinystories", "family_lopo") >= 0.75 * fidelity("fineweb", "family_lopo"),
            "retention": fidelity("tinystories", "family_lopo") / max(fidelity("fineweb", "family_lopo"), 1e-30),
        },
        "E7_lopo_support_is_nonrandom": {
            "supported": all(
                summaries[c]["support_jaccard"]["family_lopo"]
                - summaries[c]["support_jaccard"]["random"] >= 0.03
                for c in summaries
            ),
        },
        "E8_same_pair_leakage_is_limited": {
            "supported": all(
                fidelity(c, "family_inclusive") - fidelity(c, "family_lopo") <= 0.10
                for c in summaries
            ),
            "losses": {
                c: fidelity(c, "family_inclusive") - fidelity(c, "family_lopo")
                for c in summaries
            },
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Leave-one-pair-out red team of Layer-3 family channels",
        "",
        "For every query, the family and global channel rankings are rebuilt without the discovery example for that lexical pair. All other semantic-family sets are evaluated separately; their per-query median and strongest downstream fidelities are reported. This separates family reuse from same-pair recurrence.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Leave-one-pair-out channels](layer3_frozen_channel_lopo.png)",
        "",
        "| Corpus | Route | Local fidelity | Downstream fidelity | Downstream cosine | Target projection | Oracle support Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    support_names = {"pair", "family_inclusive", "family_lopo", "global_lopo", "random"}
    for corpus, summary in summaries.items():
        for name in displayed:
            row = summary[name]
            overlap = summary["support_jaccard"].get(name) if name in support_names else None
            overlap_text = "—" if overlap is None else f"{overlap:.3f}"
            lines.append(
                f"| {corpus} | {name} | {row['local_fidelity']:.3f} | "
                f"{row['downstream_fidelity']:.3f} | {row['downstream_cosine']:.3f} | "
                f"{row['target_projection']:.3f} | {overlap_text} |"
            )
    lines += [
        "",
        "## Family heterogeneity (downstream fidelity)",
        "",
        "| Corpus | Family | Pair | Family LOPO | Global LOPO | Median wrong | Best wrong | Random |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for family, row in summary["by_family"].items():
            lines.append(
                f"| {corpus} | {family} | {row['pair']:.3f} | {row['family_lopo']:.3f} | "
                f"{row['global_lopo']:.3f} | {row['wrong_family_median']:.3f} | "
                f"{row['wrong_family_best']:.3f} | {row['random']:.3f} |"
            )
    lines += [
        "",
        "A positive leave-one-pair-out result supports a family-reused native address set, but not a complete family circuit: the selected current-example channel deltas may still depend on lexical and contextual directions upstream. Comparison with the global set tests whether semantic grouping adds information beyond generally high-utility channels.",
        "The aggregate family result is heterogeneous: increasing-scale fails even in-domain, while polarity and sentiment can be matched or exceeded by the best wrong family. Family labels therefore identify useful reusable address priors, not mutually exclusive native modules.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    plot_names = ["pair", "family_inclusive", "family_lopo", "global_lopo", "wrong_family_median", "wrong_family_best", "random"]
    x = np.arange(len(plot_names))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        axis.bar(x, [summaries[corpus][name]["downstream_fidelity"] for name in plot_names])
        axis.set_xticks(x, plot_names, rotation=38, ha="right")
        axis.set_ylabel("fidelity to full MLP3 patch")
        axis.set_title(corpus)
        axis.axhline(0, color="black", linewidth=.7)
        axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
