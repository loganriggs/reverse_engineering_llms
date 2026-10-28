#!/usr/bin/env python3
"""Equal-data control for the borderline semantic-family route advantage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_rank_redteam import all_edge_projections
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_route_family_transfer import (
    address,
    evaluate_routes,
    median,
    select_route,
    shifted_routes,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_semantic_route_matched_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_route_matched_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ROUTE_MATCHED_REDTEAM.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    parent_config = json.loads(Path(config["parent_config"]).read_text())
    parent_results = json.loads(Path(config["parent_results"]).read_text())
    frozen_config = json.loads(Path(parent_config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    semantic = json.loads(Path(fixed["source_config"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    layer, rank = parent_config["layer"], parent_config["rank"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()

    fine_config = fixed["evaluation"]["fineweb"]
    source = np.load(fine_config["path"], mmap_mode="r")
    fine_rows = source[fine_config["start"]:fine_config["start"] + fine_config["rows"]]
    window_rng = np.random.default_rng(path_config["seed"])
    discovery = []
    for family, pairs in semantic["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(fine_rows, source_token, edit_position, final_position,
                                     atlas_config["contexts_per_pair"], window_rng)
            row_index, start = windows[parent_config["discovery_context_index"]]
            base_raw = np.asarray(fine_rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
            counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
            base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer)[layer]
            counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None], last_layer=layer)[layer]
            ranked = all_edge_projections(model.blocks[layer], base, counter,
                                          edit_position, final_position)
            discovery.append({"family": family, "pair": f"{source_token}->{target_token}",
                              "source_token": source_token, "target_token": target_token,
                              "scores": {address(row): float(row[0]) for row in ranked}})

    rng = np.random.default_rng(config["seed"])
    matched_routes = {}
    for heldout in discovery:
        same = [row for row in discovery if row["family"] == heldout["family"]
                and row["pair"] != heldout["pair"]]
        outside = [row for row in discovery if row["family"] != heldout["family"]]
        routes = {}
        for draw in range(config["matched_outside_draws"]):
            selected = rng.choice(len(outside), size=len(same), replace=False)
            routes[f"outside_{draw}"] = select_route([outside[int(i)] for i in selected], rank)
        matched_routes[heldout["pair"]] = routes

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        window_rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in semantic["families"].items():
            for source_token, target_token in pairs:
                pair = f"{source_token}->{target_token}"
                windows = select_windows(rows, source_token, edit_position, final_position,
                                         atlas_config["contexts_per_pair"], window_rng)
                row_index, start = windows[parent_config["evaluation_context_index"]]
                base_raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
                counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
                base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer)[layer]
                counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None], last_layer=layer)[layer]
                family_route = [tuple(edge) for edge in parent_results["route_bank"][pair]["family"]]
                routes = {"family": family_route, **matched_routes[pair]}
                for shift_name, route in shifted_routes(family_route, model.config.n_head).items():
                    routes[f"family_{shift_name}"] = route
                evaluated = evaluate_routes(model, layer, base, counter, routes,
                                            edit_position, final_position)
                family_effect = evaluated["routes"]["family"]["removal_target_projection"]
                outside_effects = [evaluated["routes"][f"outside_{draw}"]["removal_target_projection"]
                                   for draw in range(config["matched_outside_draws"])]
                shift_effects = [evaluated["routes"][f"family_shift{shift}"]["removal_target_projection"]
                                 for shift in range(1, model.config.n_head)]
                records.append({"family": family, "pair": pair, "row": int(row_index),
                                "family_effect": family_effect,
                                "outside_median": median(outside_effects),
                                "outside_effects": outside_effects,
                                "family_shift_median": median(shift_effects),
                                "advantage": family_effect - median(outside_effects)})
                print(corpus, family, pair, flush=True)
        family_summary = {}
        for family in semantic["families"]:
            selected = [row for row in records if row["family"] == family]
            family_summary[family] = {
                "family_effect": median([row["family_effect"] for row in selected]),
                "outside_median": median([row["outside_median"] for row in selected]),
                "advantage": median([row["advantage"] for row in selected]),
            }
        results[corpus] = {
            "summary": {
                "family_effect": median([row["family_effect"] for row in records]),
                "outside_median": median([row["outside_median"] for row in records]),
                "advantage": median([row["advantage"] for row in records]),
                "pair_win_rate": float(np.mean([row["advantage"] > 0 for row in records])),
                "family_shift_median": median([row["family_shift_median"] for row in records]),
            },
            "families": family_summary, "records": records,
        }

    m0 = all(result["summary"]["advantage"] >= .03 for result in results.values())
    m1 = all(result["summary"]["pair_win_rate"] >= .70 for result in results.values())
    m2 = all(abs(result["summary"]["family_effect"])
             >= 1.5 * abs(result["summary"]["family_shift_median"])
             for result in results.values())
    consistent = sum(all(results[c]["families"][family]["advantage"] > 0 for c in results)
                     for family in semantic["families"])
    hypotheses = {
        "M0_family_advantage_survives_data_matching": {"supported": m0},
        "M1_family_route_wins_most_pairs": {"supported": m1},
        "M2_family_route_is_head_specific": {"supported": m2},
        "M3_advantage_is_not_one_family": {"supported": consistent >= 3,
                                            "families_positive_both": consistent},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))
    lines = ["# Data-matched red-team of semantic-family routes", "",
             "Every same-family route is compared with sixteen outside-family routes learned from the same number of discovery examples and evaluated under the same finite patch.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Family effect | Matched outside | Advantage | Pair win rate | Family head-shift |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(f"| {corpus} | {row['family_effect']:.3f} | {row['outside_median']:.3f} | {row['advantage']:.3f} | {row['pair_win_rate']:.3f} | {row['family_shift_median']:.3f} |")
    lines += ["", "| Family | FineWeb advantage | TinyStories advantage |", "|---|---:|---:|"]
    for family in semantic["families"]:
        lines.append(f"| {family} | {results['fineweb']['families'][family]['advantage']:.3f} | {results['tinystories']['families'][family]['advantage']:.3f} |")
    lines += ["", "A positive matched advantage is preliminary family-level route reuse. With only three or four lexical pairs per family, it remains vulnerable to token-frequency and syntactic-position confounds and requires new pairs for confirmation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
