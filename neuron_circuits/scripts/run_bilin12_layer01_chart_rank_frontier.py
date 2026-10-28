#!/usr/bin/env python3
"""Rank-versus-chart-bank frontier for prospective suffix-Fisher routing."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import midpoint_product_jvp, write_fidelity
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin12_layer01_suffix_fisher_chart_router import suffix_jvp_batch, trace_from_x0
from run_bilin18_layer01_native_rotation_tournament import make_rotation, rotate, selected_rotated
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin12_layer01_chart_rank_frontier.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_chart_rank_frontier.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_CHART_RANK_FRONTIER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_chart_rank_frontier.png"


def chart_program_precomputed(full, router, decoder, chart_decoder, rank, permutations=None):
    if permutations is None:
        router_coordinates, full_coordinates = router, full
        router_write = F.linear(router_coordinates, decoder)
        indices = (router_coordinates * (router_write @ decoder)).abs().topk(rank).indices
        router_patch = selected_delta(router, indices)
        full_patch = selected_delta(full, indices)
        local = write_fidelity(decoder, router, router_patch)
    else:
        router_coordinates = rotate(router, permutations)
        full_coordinates = rotate(full, permutations)
        router_write = F.linear(router_coordinates, chart_decoder)
        indices = (router_coordinates * (router_write @ chart_decoder)).abs().topk(rank).indices
        router_selected = selected_delta(router_coordinates, indices)
        full_patch = selected_rotated(full, indices, permutations)
        router_patch = selected_rotated(router, indices, permutations)
        local = write_fidelity(chart_decoder, router_coordinates, router_selected)
    return full_patch, router_patch, local


def subset_rows(seed_names, counts):
    return {count: list(combinations(seed_names, count)) for count in counts}


def main():
    red = json.loads(CONFIG.read_text())
    fisher = json.loads(Path(red["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    parent = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    seeds = parent["rotation_seeds"]
    seed_names = [str(seed) for seed in seeds]
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]] for corpus, spec in config["evaluation"].items()}
    layers = config["layers"]; ranks = red["ranks"]
    edit_position, position = config["edit_position"], config["prediction_position"]
    rotations = {layer: {seed: make_rotation(model.blocks[layer].mlp.Down.weight.shape[1], config["rotation_rounds"], seed + layer, "cuda") for seed in seeds} for layer in layers}
    rotated_decoders = {layer: {seed: rotate(model.blocks[layer].mlp.Down.weight.float(), rotations[layer][seed]) for seed in seeds} for layer in layers}
    records = {str(layer): [] for layer in layers}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                windows = select_windows(rows, source_token, edit_position, position, config["evaluation_contexts_per_pair"], rng)
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy(); changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers)); counter = forward_layers_native(model, changed_ids, max(layers))
                    midpoint_trace = None
                    for layer in layers:
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        tangent, point, _ = midpoint_product_jvp(model, base_ids, changed_ids, layer, position, router["path_point"])
                        if midpoint_trace is None:
                            with torch.no_grad(): midpoint_trace = trace_from_x0(model, point, max(layers))
                        actual_patches = {}; tangent_patches = {}; names = []
                        for rank in ranks:
                            name = f"{rank}:native"; names.append(name)
                            actual, tangent_patch, _ = chart_program_precomputed(full, tangent, decoder, decoder, rank)
                            actual_patches[name] = actual; tangent_patches[name] = tangent_patch
                            for seed in seeds:
                                name = f"{rank}:{seed}"; names.append(name)
                                actual, tangent_patch, _ = chart_program_precomputed(full, tangent, decoder, rotated_decoders[layer][seed], rank, rotations[layer][seed])
                                actual_patches[name] = actual; tangent_patches[name] = tangent_patch
                        write_tangents = [F.linear(tangent, decoder)] + [F.linear(tangent_patches[name], decoder) for name in names]
                        logits, derivatives, _, _, _ = suffix_jvp_batch(model, layer, midpoint_trace[layer], write_tangents, position)
                        predicted = {name: fisher_fidelity(logits[0], derivatives[0], derivatives[index + 1])[0] for index, name in enumerate(names)}
                        with torch.no_grad(): effects = evaluate_deltas(model, layer, base[layer], {name: actual_patches[name] for name in names}, full, position)
                        charts = {str(rank): {chart: {"predicted": predicted[f"{rank}:{chart}"], "actual": effects[f"{rank}:{chart}"]} for chart in ["native"] + seed_names} for rank in ranks}
                        records[str(layer)].append({"corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "context": context_index, "charts": charts})
                print("chart-rank frontier", corpus, family, source_token, target_token, flush=True)

    subsets = subset_rows(seed_names, red["rotation_counts"])
    summaries = {str(layer): {} for layer in layers}
    subset_details = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(config["evaluation"]):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {}; subset_details[str(layer)][corpus] = {}
            for rank_index, rank in enumerate(ranks):
                summaries[str(layer)][corpus][str(rank)] = {}; subset_details[str(layer)][corpus][str(rank)] = {}
                for count in red["rotation_counts"]:
                    rows = []
                    for subset in subsets[count]:
                        candidates = ("native",) + subset
                        routed = []; oracle = []; rotated = []
                        for record in selected:
                            chart = record["charts"][str(rank)]
                            choice = max(candidates, key=lambda name: chart[name]["predicted"])
                            routed.append(chart[choice]["actual"]); oracle.append(max(chart[name]["actual"] for name in candidates)); rotated.append(choice != "native")
                        rows.append({"subset": list(subset), "routed": median(routed), "oracle": median(oracle), "rotated_fraction": float(np.mean(rotated))})
                    subset_details[str(layer)][corpus][str(rank)][str(count)] = rows
                    summaries[str(layer)][corpus][str(rank)][str(count)] = {
                        "routed_median_subset": median([row["routed"] for row in rows]),
                        "routed_minimum_subset": min(row["routed"] for row in rows),
                        "routed_maximum_subset": max(row["routed"] for row in rows),
                        "oracle_median_subset": median([row["oracle"] for row in rows]),
                        "maximum_oracle_gap": max(row["oracle"] - row["routed"] for row in rows),
                        "rotated_fraction_median_subset": median([row["rotated_fraction"] for row in rows]),
                        "payload_coordinates": rank,
                        "chart_id_bits": int(math.ceil(math.log2(1 + count))) if count else 0,
                        "router_chart_evaluations": 1 + count,
                    }
                # Pair-clustered primary curve for the complete seven-rotation bank.
                advantage, interval = pair_cluster_interval(
                    selected,
                    lambda row, rank=rank: (
                        row["charts"][str(rank)][max(["native"] + seed_names, key=lambda name: row["charts"][str(rank)][name]["predicted"])]["actual"]
                        - row["charts"][str(rank)]["native"]["actual"]
                    ), red["bootstrap_samples"], red["bootstrap_seed"] + layer * 1000 + corpus_index * 100 + rank_index,
                )
                summaries[str(layer)][corpus][str(rank)]["seven_minus_native_pair_median"] = advantage
                summaries[str(layer)][corpus][str(rank)]["seven_minus_native_pair_ci"] = interval

    def cell(layer, corpus, rank, count): return summaries[str(layer)][corpus][str(rank)][str(count)]
    p0 = all(cell(l, c, r, m)["maximum_oracle_gap"] <= .03 for l in layers for c in config["evaluation"] for r in ranks for m in red["rotation_counts"])
    p1 = True; capture = {}
    for l in layers:
        capture[str(l)] = {}
        for c in config["evaluation"]:
            native = cell(l, c, 256, 0)["routed_median_subset"]; two = cell(l, c, 256, 2)["routed_median_subset"]; seven = cell(l, c, 256, 7)["routed_median_subset"]
            fraction = (two - native) / max(seven - native, 1e-30); capture[str(l)][c] = fraction; p1 = p1 and fraction >= .8
    p2 = all(cell(l, c, 128, 7)["routed_median_subset"] >= cell(l, c, 256, 0)["routed_median_subset"] - .02 for l in layers for c in config["evaluation"])
    p3 = all(cell(l, c, 64, 7)["routed_median_subset"] >= cell(l, c, 256, 0)["routed_median_subset"] - .02 for l in layers for c in config["evaluation"])
    p4 = all(cell(l, c, 256, 7)["routed_median_subset"] - cell(l, c, 256, 4)["routed_median_subset"] <= .02 for l in layers for c in config["evaluation"])
    p5 = all(cell(l, c, 256, m)["routed_maximum_subset"] - cell(l, c, 256, m)["routed_minimum_subset"] <= .05 for l in layers for c in config["evaluation"] for m in (2, 4))
    p6 = all((cell(1, c, 128, 7)["routed_median_subset"] - cell(1, c, 128, 0)["routed_median_subset"]) - (cell(0, c, 128, 7)["routed_median_subset"] - cell(0, c, 128, 0)["routed_median_subset"]) >= .03 for c in config["evaluation"])
    p7 = all(abs((cell(l, "fineweb", r, 7)["routed_median_subset"] - cell(l, "fineweb", r, 0)["routed_median_subset"]) - (cell(l, "tinystories", r, 7)["routed_median_subset"] - cell(l, "tinystories", r, 0)["routed_median_subset"])) <= .1 for l in layers for r in ranks)
    hypotheses = {
        "P0_suffix_router_stays_near_the_subset_oracle": {"supported": p0},
        "P1_two_rotation_atlas_captures_most_of_the_gain": {"supported": p1, "fractions": capture},
        "P2_seven_charts_halve_the_coordinate_budget": {"supported": p2},
        "P3_seven_charts_quarter_the_coordinate_budget": {"supported": p3},
        "P4_returns_diminish_after_four_rotations": {"supported": p4},
        "P5_chart_subset_choice_is_robust": {"supported": p5},
        "P6_adaptive_chart_gain_increases_with_depth": {"supported": p6},
        "P7_chart_gain_transfers_ood": {"supported": p7},
        "P8_router_never_needs_endpoint_outcomes": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "subset_details": subset_details, "records": records}, indent=2, allow_nan=False))
    lines = ["# Rank-versus-chart-bank frontier", "", "Native plus random rotated product charts are routed using the midpoint suffix-Fisher contraction. All subsets of sizes 1, 2, and 4 from the seven rotations are enumerated; size 0 is native-only and size 7 is the complete bank. Finite effects are inspected only after chart choice is frozen.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Chart-rank frontier](bilin12_layer01_chart_rank_frontier.png)", "", "| Layer | Corpus | Rank | Native / median M1 / M2 / M4 / M7 routed fidelity | M7−native pair median (95% CI) | M2/M4 subset ranges |", "|---:|---|---:|---:|---:|---:|"]
    for l in layers:
        for c in config["evaluation"]:
            for r in ranks:
                values = "/".join(f"{cell(l,c,r,m)['routed_median_subset']:.3f}" for m in red["rotation_counts"]); row = summaries[str(l)][c][str(r)]; ci = row["seven_minus_native_pair_ci"]
                range2 = cell(l,c,r,2)["routed_maximum_subset"] - cell(l,c,r,2)["routed_minimum_subset"]; range4 = cell(l,c,r,4)["routed_maximum_subset"] - cell(l,c,r,4)["routed_minimum_subset"]
                lines.append(f"| {l} | {c} | {r} | {values} | {row['seven_minus_native_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | {range2:.3f}/{range4:.3f} |")
    lines += ["", "Complexity is reported in separate currencies: rank K is the per-example coordinate payload, `ceil(log2(1+M))` bits identify one of native plus M rotated charts, and the router evaluates 1+M suffix contractions. Stored chart definitions are an additional global cost and are not hidden inside K. A rotated payload remains dense in native coordinates."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (l, c) in zip(axes.flat, [(l,c) for l in layers for c in config["evaluation"]]):
        for m in red["rotation_counts"]: axis.plot(ranks, [cell(l,c,r,m)["routed_median_subset"] for r in ranks], marker="o", label=f"M={m}")
        axis.set_xscale("log", base=2); axis.set_ylim(0, 1.02); axis.set_title(f"Layer {l} / {c}"); axis.set_xlabel("coordinates K"); axis.set_ylabel("median Fisher fidelity"); axis.grid(alpha=.25); axis.legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
