#!/usr/bin/env python3
"""Discover fixed attention-edge routes on disjoint FineWeb contexts and transfer them."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer8_router_causal_validation import exact_kl, suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import (
    edge_mask,
    median,
    output_with_attention,
    pearson,
    selected_attention_delta,
)
from run_bilin18_layer8_semantic_attention_rank_redteam import all_edge_projections
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_attention_frozen_routes.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_frozen_routes.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_FROZEN_ROUTES.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_frozen_routes.png"


def edge_address(row):
    return int(row[1]), int(row[2]), int(row[3])


@torch.inference_mode()
def discover_routes(model, rows, families, config, atlas_config, path_seed,
                    edit_position, final_position):
    rng = np.random.default_rng(path_seed)
    values = {layer: collections.defaultdict(list) for layer in config["layers"]}
    discovery_rows = []
    for family, pairs in families.items():
        for source_token, target_token in pairs:
            windows = select_windows(
                rows, source_token, edit_position, final_position,
                atlas_config["contexts_per_pair"], rng,
            )
            row_index, start = windows[config["discovery_context_index"]]
            base_raw = np.asarray(rows[row_index, start:start + final_position + 1], dtype=np.int64).copy()
            counterfactual_raw = base_raw.copy()
            counterfactual_raw[edit_position] = target_token
            base_layers = forward_layers(
                model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=max(config["layers"]),
            )
            counterfactual_layers = forward_layers(
                model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=max(config["layers"]),
            )
            for layer in config["layers"]:
                ranked = all_edge_projections(
                    model.blocks[layer], base_layers[layer], counterfactual_layers[layer],
                    edit_position, final_position,
                )
                for projection, head, destination, source in ranked:
                    values[layer][(head, destination, source)].append(projection)
            discovery_rows.append({
                "family": family,
                "pair": f"{source_token}->{target_token}",
                "row": int(row_index),
                "source_start": int(start),
            })
            print("discover", family, source_token, "->", target_token, flush=True)

    discovered = {}
    for layer in config["layers"]:
        summaries = []
        for address, projections in values[layer].items():
            projections = np.asarray(projections, dtype=np.float64)
            summaries.append({
                "edge": list(address),
                "mean_absolute_projection": float(np.mean(np.abs(projections))),
                "absolute_mean_projection": float(abs(np.mean(projections))),
                "mean_projection": float(np.mean(projections)),
            })
        magnitude = sorted(summaries, key=lambda row: row["mean_absolute_projection"], reverse=True)
        signed = sorted(summaries, key=lambda row: row["absolute_mean_projection"], reverse=True)
        discovered[layer] = {"magnitude": magnitude, "signed": signed}
    return discovered, discovery_rows


def route_definitions(discovered, ranked, ranks, heads):
    routes = {}
    for rank in ranks:
        routes[f"frozen{rank}"] = [tuple(row["edge"]) for row in discovered["magnitude"][:rank]]
    routes["signed64"] = [tuple(row["edge"]) for row in discovered["signed"][:64]]
    routes["oracle64"] = [edge_address(row) for row in ranked[:64]]
    frozen64 = routes["frozen64"]
    for shift in range(1, heads):
        routes[f"head_shift{shift}"] = [
            ((head + shift) % heads, destination, source)
            for head, destination, source in frozen64
        ]
    return routes


@torch.inference_mode()
def analyze_layer(model, layer, base, counterfactual, discovered, config,
                  edit_position, final_position):
    block = model.blocks[layer]
    ranked = all_edge_projections(
        block, base, counterfactual, edit_position, final_position,
    )
    routes = route_definitions(discovered, ranked, config["ranks"], model.config.n_head)
    score = {edge_address(row): row[0] for row in ranked}
    deltas, local = {}, {}
    for name, selected in routes.items():
        local[name] = float(sum(score[edge] for edge in selected))
        deltas[name] = selected_attention_delta(
            block, base, counterfactual,
            edge_mask(
                model.config.n_head, base["output"].shape[1], selected,
                base["output"].device,
            ),
        )

    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    indices = {}
    for name, delta in deltas.items():
        indices[name] = (len(outputs), len(outputs) + 1)
        outputs += [
            output_with_attention(block, base, base["attention_write"] + delta),
            output_with_attention(
                block, counterfactual, counterfactual["attention_write"] - delta,
            ),
        ]
        x0s += [base["x0"], counterfactual["x0"]]
        buses += [base["value_bus"], counterfactual["value_bus"]]
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    target = counterfactual_logits - base_logits
    energy = fisher_inner(base_logits, target, target)
    full_kl = exact_kl(base_logits, counterfactual_logits)
    results = {}
    for name, (insertion_index, removal_index) in indices.items():
        insertion = logits[insertion_index] - base_logits
        removal = counterfactual_logits - logits[removal_index]
        result = {"edge_count": len(routes[name]), "local_projection": local[name]}
        for direction, effect, changed in (
            ("insertion", insertion, logits[insertion_index]),
            ("removal", removal, logits[removal_index]),
        ):
            fidelity, cosine, _ = fisher_fidelity(base_logits, target, effect)
            result[direction] = {
                "target_projection": fisher_inner(base_logits, effect, target) / max(energy, 1e-30),
                "fisher_fidelity": fidelity,
                "fisher_cosine": cosine,
                "kl_ratio_to_full": exact_kl(base_logits, changed) / max(full_kl, 1e-30),
            }
        results[name] = result
    return {"full_kl": full_kl, "routes": results}


def summarize(records, layers, route_names):
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        selected = [record["layers"][layer_index] for record in records]
        routes = {}
        for name in route_names:
            routes[name] = {
                "local_projection": median([row["routes"][name]["local_projection"] for row in selected]),
                "insertion": {
                    key: median([row["routes"][name]["insertion"][key] for row in selected])
                    for key in ("target_projection", "fisher_fidelity", "fisher_cosine", "kl_ratio_to_full")
                },
                "removal": {
                    key: median([row["routes"][name]["removal"][key] for row in selected])
                    for key in ("target_projection", "fisher_fidelity", "fisher_cosine", "kl_ratio_to_full")
                },
            }
        head_shifts = [routes[f"head_shift{shift}"]["removal"]["target_projection"] for shift in range(1, 9)]
        routes["head_shift_median"] = {"removal_target_projection": median(head_shifts)}
        layer_rows.append({"layer": layer, "routes": routes})

    insertion = [
        record["layers"][layer_index]["routes"]["frozen64"]["insertion"]["target_projection"]
        for record in records for layer_index in range(len(layers))
    ]
    removal = [
        record["layers"][layer_index]["routes"]["frozen64"]["removal"]["target_projection"]
        for record in records for layer_index in range(len(layers))
    ]
    return {
        "layers": layer_rows,
        "frozen64_insertion_to_removal_correlation": pearson(insertion, removal),
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    rank_config = json.loads(Path(config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]

    fine_config = fixed["evaluation"]["fineweb"]
    fine_source = np.load(fine_config["path"], mmap_mode="r")
    fine_rows = fine_source[fine_config["start"]:fine_config["start"] + fine_config["rows"]]
    discovered, discovery_rows = discover_routes(
        model, fine_rows, parent["families"], config, atlas_config,
        path_config["seed"], edit_position, final_position,
    )
    route_names = (
        [f"frozen{rank}" for rank in config["ranks"]]
        + ["signed64", "oracle64"]
        + [f"head_shift{shift}" for shift in range(1, model.config.n_head)]
    )
    results, records_by_corpus = {}, {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                row_index, start = windows[config["evaluation_context_index"]]
                base_raw = np.asarray(
                    rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                ).copy()
                counterfactual_raw = base_raw.copy()
                counterfactual_raw[edit_position] = target_token
                base_layers = forward_layers(
                    model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=max(config["layers"]),
                )
                counterfactual_layers = forward_layers(
                    model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=max(config["layers"]),
                )
                layer_rows = [
                    analyze_layer(
                        model, layer, base_layers[layer], counterfactual_layers[layer],
                        discovered[layer], config, edit_position, final_position,
                    )
                    for layer in config["layers"]
                ]
                records.append({
                    "family": family,
                    "pair": f"{source_token}->{target_token}",
                    "row": int(row_index),
                    "source_start": int(start),
                    "layers": layer_rows,
                })
                print("evaluate", corpus, family, source_token, "->", target_token, flush=True)
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records, config["layers"], route_names)

    selected_index = max(
        range(len(config["layers"])),
        key=lambda index: abs(
            results["fineweb"]["layers"][index]["routes"]["frozen64"]["removal"]["target_projection"]
        ),
    )
    selected_layer = config["layers"][selected_index]
    frozen_oracle_fraction, frozen_shift_ratio = {}, {}
    for corpus in results:
        routes = results[corpus]["layers"][selected_index]["routes"]
        frozen = abs(routes["frozen64"]["removal"]["target_projection"])
        oracle = abs(routes["oracle64"]["removal"]["target_projection"])
        shifted = abs(routes["head_shift_median"]["removal_target_projection"])
        frozen_oracle_fraction[corpus] = frozen / max(oracle, 1e-30)
        frozen_shift_ratio[corpus] = frozen / max(shifted, 1e-30)
    fine_projection = abs(
        results["fineweb"]["layers"][selected_index]["routes"]["frozen64"]["removal"]["target_projection"]
    )
    ood_retention = abs(
        results["tinystories"]["layers"][selected_index]["routes"]["frozen64"]["removal"]["target_projection"]
    ) / max(fine_projection, 1e-30)
    hypotheses = {
        "S1_frozen_route_is_causal": {
            "supported": all(
                results[corpus]["layers"][selected_index]["routes"]["frozen64"]["removal"]["target_projection"] >= 0.10
                for corpus in results
            ),
        },
        "S2_frozen_route_recovers_half_oracle": {
            "supported": all(value >= 0.50 for value in frozen_oracle_fraction.values()),
            "fractions": frozen_oracle_fraction,
        },
        "S3_frozen_heads_beat_shifted_controls": {
            "supported": all(value >= 2 for value in frozen_shift_ratio.values()),
            "ratios": frozen_shift_ratio,
        },
        "S4_larger_frozen_route_adds_effect": {
            "supported": all(
                results[corpus]["layers"][selected_index]["routes"]["frozen128"]["removal"]["target_projection"]
                >= results[corpus]["layers"][selected_index]["routes"]["frozen64"]["removal"]["target_projection"]
                for corpus in results
            ),
        },
        "S5_frozen_route_transfers_ood": {
            "supported": ood_retention >= 0.50,
            "ood_retention": ood_retention,
        },
        "S6_frozen_route_is_cross_world_stable": {
            "supported": all(
                results[corpus]["frozen64_insertion_to_removal_correlation"] >= 0.70
                for corpus in results
            ),
        },
    }
    compact_discovery = {
        str(layer): {
            family: rows[:max(config["ranks"])]
            for family, rows in discovered[layer].items()
        }
        for layer in config["layers"]
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "selected_layer": selected_layer,
        "hypotheses": hypotheses,
        "discovery_records": discovery_rows,
        "discovered_routes": compact_discovery,
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Frozen semantic-attention route transfer",
        "",
        "Fixed edge addresses are learned only from the unused second FineWeb context for each lexical pair. Edges are ranked by mean absolute local projection, frozen, and evaluated on the first FineWeb contexts and on TinyStories. `oracle64` is selected separately per evaluation example and is only a ceiling. Eight controls preserve every frozen edge's source and destination while cyclically shifting its head.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Frozen semantic attention routes](layer8_semantic_attention_frozen_routes.png)",
        "",
        f"FineWeb-selected layer: **{selected_layer}**.",
        "",
        "| Corpus | Layer | Frozen 8/16/32/64/128 removal | Signed-64 | Oracle-64 | Head-shift median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for row in result["layers"]:
            routes = row["routes"]
            nested = "/".join(
                f"{routes[f'frozen{rank}']['removal']['target_projection']:.3f}"
                for rank in config["ranks"]
            )
            lines.append(
                f"| {corpus} | {row['layer']} | {nested} | "
                f"{routes['signed64']['removal']['target_projection']:.3f} | "
                f"{routes['oracle64']['removal']['target_projection']:.3f} | "
                f"{routes['head_shift_median']['removal_target_projection']:.3f} |"
            )
    lines += [
        "",
        "| Corpus | Frozen/oracle at selected layer | Frozen/head-shift | Insertion→removal correlation |",
        "|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        lines.append(
            f"| {corpus} | {frozen_oracle_fraction[corpus]:.3f} | "
            f"{frozen_shift_ratio[corpus]:.3f} | "
            f"{result['frozen64_insertion_to_removal_correlation']:.3f} |"
        )
    lines += [
        "",
        f"TinyStories retention relative to FineWeb at the selected layer: `{ood_retention:.3f}`.",
        "",
        "This tests recurrence of edge addresses, not recurrence of a semantic activation direction. A fixed route can broadcast many lexical changes without representing their shared human category.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for layer_index, layer in enumerate(config["layers"]):
            axis.plot(
                config["ranks"],
                [
                    results[corpus]["layers"][layer_index]["routes"][f"frozen{rank}"]["removal"]["target_projection"]
                    for rank in config["ranks"]
                ],
                marker="o",
                label=f"layer {layer}",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(config["ranks"], config["ranks"])
        axis.set_title(corpus)
        axis.set_xlabel("frozen FineWeb edge addresses")
        axis.set_ylabel("downstream target projection")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
