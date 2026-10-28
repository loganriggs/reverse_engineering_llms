#!/usr/bin/env python3
"""Prospective new-pair validation of semantic-family attention routes."""

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


CONFIG = Path("configs/bilin18_layer8_semantic_route_prospective.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_route_prospective.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ROUTE_PROSPECTIVE.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    family_config = json.loads(Path(config["source_config"]).read_text())
    frozen_config = json.loads(Path(family_config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    semantic = json.loads(Path(fixed["source_config"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    layer, rank = config["layer"], config["rank"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()

    # Freeze family/global routes using only the original-pair discovery panel.
    fine_config = fixed["evaluation"]["fineweb"]
    source = np.load(fine_config["path"], mmap_mode="r")
    fine_rows = source[fine_config["start"]:fine_config["start"] + fine_config["rows"]]
    rng = np.random.default_rng(path_config["seed"])
    discovery = []
    for family, pairs in semantic["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(fine_rows, source_token, edit_position, final_position,
                                     atlas_config["contexts_per_pair"], rng)
            row_index, start = windows[family_config["discovery_context_index"]]
            base_raw = np.asarray(fine_rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
            counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
            base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer)[layer]
            counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None], last_layer=layer)[layer]
            ranked = all_edge_projections(model.blocks[layer], base, counter,
                                          edit_position, final_position)
            discovery.append({"family": family,
                              "scores": {address(row): float(row[0]) for row in ranked}})
    family_routes = {
        family: select_route([row for row in discovery if row["family"] == family], rank)
        for family in semantic["families"]
    }
    global_route = select_route(discovery, rank)

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        window_rng = np.random.default_rng(config["seed"] + corpus_index)
        records = []
        for family, pairs in config["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(rows, source_token, edit_position, final_position,
                                         config["contexts_per_pair"], window_rng)
                for context_index, (row_index, start) in enumerate(windows):
                    base_raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
                    counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
                    base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer)[layer]
                    counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None], last_layer=layer)[layer]
                    routes = {f"family_{name}": route for name, route in family_routes.items()}
                    routes["global"] = global_route
                    for shift_name, route in shifted_routes(family_routes[family], model.config.n_head).items():
                        routes[f"correct_{shift_name}"] = route
                    evaluated = evaluate_routes(model, layer, base, counter, routes,
                                                edit_position, final_position)
                    correct = evaluated["routes"][f"family_{family}"]["removal_target_projection"]
                    wrong = [evaluated["routes"][f"family_{name}"]["removal_target_projection"]
                             for name in family_routes if name != family]
                    shifts = [evaluated["routes"][f"correct_shift{shift}"]["removal_target_projection"]
                              for shift in range(1, model.config.n_head)]
                    records.append({
                        "family": family, "pair": f"{source_token}->{target_token}",
                        "context": context_index, "row": int(row_index),
                        "correct": correct, "wrong_median": median(wrong),
                        "advantage": correct - median(wrong),
                        "global": evaluated["routes"]["global"]["removal_target_projection"],
                        "head_shift_median": median(shifts),
                        "oracle": evaluated["routes"]["oracle"]["removal_target_projection"],
                    })
                print(corpus, family, source_token, target_token, flush=True)
        summary = {key: median([row[key] for row in records]) for key in
                   ("correct", "wrong_median", "advantage", "global",
                    "head_shift_median", "oracle")}
        summary["win_rate"] = float(np.mean([row["advantage"] > 0 for row in records]))
        families = {}
        for family in config["new_families"]:
            selected = [row for row in records if row["family"] == family]
            families[family] = {"correct": median([row["correct"] for row in selected]),
                                "wrong_median": median([row["wrong_median"] for row in selected]),
                                "advantage": median([row["advantage"] for row in selected]),
                                "win_rate": float(np.mean([row["advantage"] > 0 for row in selected]))}
        results[corpus] = {"summary": summary, "families": families, "records": records}

    p0 = all(r["summary"]["advantage"] >= .03 for r in results.values())
    p1 = all(r["summary"]["win_rate"] >= .70 for r in results.values())
    p2 = all(r["summary"]["correct"] >= .05
             and abs(r["summary"]["correct"]) >= 1.5 * abs(r["summary"]["head_shift_median"])
             for r in results.values())
    p3 = all(abs(r["summary"]["global"] - r["summary"]["correct"]) <= .03
             and r["summary"]["advantage"] < .03 for r in results.values())
    p4 = all(r["summary"]["oracle"]
             >= max(r["summary"]["correct"], r["summary"]["global"]) + .05
             for r in results.values())
    hypotheses = {
        "P0_correct_semantic_route_transfers_to_new_pairs": {"supported": p0},
        "P1_correct_family_wins_most_new_examples": {"supported": p1},
        "P2_correct_route_is_causal_and_head_specific": {"supported": p2},
        "P3_generic_broadcast_is_sufficient": {"supported": p3},
        "P4_oracle_gap_persists": {"supported": p4},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses,
        "frozen_routes": {name: [list(edge) for edge in route]
                          for name, route in {**family_routes, "global": global_route}.items()},
        "results": results,
    }, indent=2))
    lines = ["# Prospective semantic-route transfer to new lexical pairs", "",
             "Routes learned only from the original pairs are evaluated on seventeen newly declared substitutions, three contexts per pair, on FineWeb and TinyStories.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Correct family | Wrong-family median | Advantage | Win rate | Global | Head shift | Oracle |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(f"| {corpus} | {row['correct']:.3f} | {row['wrong_median']:.3f} | {row['advantage']:.3f} | {row['win_rate']:.3f} | {row['global']:.3f} | {row['head_shift_median']:.3f} | {row['oracle']:.3f} |")
    lines += ["", "| Family | FineWeb advantage/win | TinyStories advantage/win |", "|---|---:|---:|"]
    for family in config["new_families"]:
        fine = results["fineweb"]["families"][family]
        tiny = results["tinystories"]["families"][family]
        lines.append(f"| {family} | {fine['advantage']:.3f}/{fine['win_rate']:.2f} | {tiny['advantage']:.3f}/{tiny['win_rate']:.2f} |")
    lines += ["", "The semantic labels and all new pairs were frozen before evaluation. Success identifies reusable routing associated with the declared family; it still does not imply that an edge or head contains the concept independently of its value/state content."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
