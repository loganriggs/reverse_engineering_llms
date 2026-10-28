#!/usr/bin/env python3
"""Diagnose why frozen Taylor channel selection failed on random token edits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_conditional_coalition_selector import jaccard, local_scores
from run_bilin12_layer01_directional_order_frontier import derivative_frontier, taylor_sum
from run_bilin12_layer01_dynamic_alignment_replication import final_product_delta, forward_layers_native, median
from run_bilin12_layer01_frozen_taylor_behavior import product_from_x0, sample_edits
from run_bilin12_layer01_directional_taylor_selector import cosine
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_taylor_failure_diagnosis.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_taylor_failure_diagnosis.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_TAYLOR_FAILURE_DIAGNOSIS.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_taylor_failure_diagnosis.png"


def spearman(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() < 1e-30 or right.std() < 1e-30:
        return float("nan")
    left_rank = left.argsort(kind="stable").argsort(kind="stable").astype(np.float64)
    right_rank = right.argsort(kind="stable").argsort(kind="stable").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def write_fidelity(decoder, full, indices):
    selected = torch.zeros_like(full); selected[indices] = full[indices]
    target, prediction = F.linear(full, decoder), F.linear(selected, decoder)
    return float(1 - (target - prediction).square().sum() / target.square().sum().clamp_min(1e-30))


def normalized_token_chord(model, source, target):
    dimension = model.config.n_embd
    values = model.embedding.weight[[source, target]].float()
    values = F.rms_norm(values, (dimension,))
    return float((values[1] - values[0]).norm())


def main():
    red = json.loads(CONFIG.read_text())
    behavior_config = json.loads(Path(red["source_config"]).read_text())
    behavior = json.loads(Path(red["source_results"]).read_text())
    parent = json.loads(Path(behavior_config["source_config"]).read_text())
    semantic = json.loads(Path(red["semantic_pairs"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in parent["evaluation"].items()
    }
    behavior_lookup = {
        (layer, row["corpus"], row["site"]): row
        for layer, rows in behavior["records"].items() for row in rows
    }
    records = {str(layer): [] for layer in parent["layers"]}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        edits = sample_edits(
            rows, behavior_config["sites_per_corpus"], behavior_config["sampling_seed"] + corpus_index,
            behavior_config["edit_position"], behavior_config["sequence_length"],
        )
        for site_index, (row_index, target_token) in enumerate(edits):
            raw = np.asarray(rows[row_index, :behavior_config["sequence_length"]], dtype=np.int64).copy()
            changed = raw.copy(); source_token = int(raw[behavior_config["edit_position"]])
            changed[behavior_config["edit_position"]] = target_token
            base_ids = torch.as_tensor(raw, device="cuda")[None]
            endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
                endpoint_trace = forward_layers_native(model, endpoint_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
            direction = endpoint_x0 - base_x0
            chord = float(direction.norm())
            for layer in parent["layers"]:
                function = lambda value, layer=layer: product_from_x0(
                    model, value, layer, behavior_config["prediction_position"],
                )
                with torch.enable_grad():
                    derivatives, functions = derivative_frontier(
                        function, base_x0, direction, red["maximum_order"],
                    )
                    explicit_midpoint = functions[1](base_x0 + .5 * direction)
                derivatives = [value.detach().float() for value in derivatives]
                explicit_midpoint = explicit_midpoint.detach().float()
                full = final_product_delta(
                    model, layer, base_trace[layer], endpoint_trace[layer], behavior_config["prediction_position"],
                )
                decoder = model.blocks[layer].mlp.Down.weight.float()
                endpoint_local = local_scores(full, decoder).topk(red["rank"]).indices
                endpoint_magnitude = full.abs().topk(red["rank"]).indices
                orders = {}
                for order in range(1, red["maximum_order"] + 1):
                    vector = taylor_sum(derivatives[:order], .5, integral=False)
                    local_indices = local_scores(vector, decoder).topk(red["rank"]).indices
                    magnitude_indices = vector.abs().topk(red["rank"]).indices
                    orders[str(order)] = {
                        "midpoint_cosine": cosine(vector, explicit_midpoint),
                        "endpoint_cosine": cosine(vector, full),
                        "local_support_jaccard": jaccard(local_indices.cpu().tolist(), endpoint_local.cpu().tolist()),
                        "magnitude_support_jaccard": jaccard(magnitude_indices.cpu().tolist(), endpoint_magnitude.cpu().tolist()),
                        "local_write_fidelity": write_fidelity(decoder, full, local_indices),
                        "magnitude_write_fidelity": write_fidelity(decoder, full, magnitude_indices),
                    }
                behavior_row = behavior_lookup[(str(layer), corpus, site_index)]
                records[str(layer)].append({
                    "corpus": corpus, "site": site_index, "source_token": source_token,
                    "target_token": target_token, "chord": chord,
                    "explicit_midpoint_endpoint_cosine": cosine(explicit_midpoint, full),
                    "orders": orders,
                    "behavior_kl_recovery": 1 - behavior_row["metrics"][f"taylor_local_{red['rank']}"]["kl_error"] / max(behavior_row["base_kl"], 1e-30),
                })
            if (site_index + 1) % 16 == 0:
                print("diagnosis", corpus, site_index + 1, "/", len(edits), flush=True)

    semantic_chords = [
        normalized_token_chord(model, int(source), int(target))
        for pairs in semantic["new_families"].values() for source, target in pairs
    ]
    summaries = {str(layer): {} for layer in parent["layers"]}
    for layer in parent["layers"]:
        frozen = str(behavior_config["frozen_orders"][str(layer)])
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {
                "sites": len(selected),
                "median_chord": median([row["chord"] for row in selected]),
                "median_explicit_midpoint_endpoint_cosine": median([row["explicit_midpoint_endpoint_cosine"] for row in selected]),
                "orders": {str(order): {
                    key: median([row["orders"][str(order)][key] for row in selected])
                    for key in ("midpoint_cosine", "endpoint_cosine", "local_support_jaccard", "magnitude_support_jaccard", "local_write_fidelity", "magnitude_write_fidelity")
                } for order in range(1, red["maximum_order"] + 1)},
                "chord_behavior_spearman": spearman(
                    [row["chord"] for row in selected], [row["behavior_kl_recovery"] for row in selected],
                ),
                "support_behavior_spearman": spearman(
                    [row["orders"][frozen]["local_support_jaccard"] for row in selected],
                    [row["behavior_kl_recovery"] for row in selected],
                ),
            }
    semantic_median = median(semantic_chords)
    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    hypotheses = {
        "D0_sites_reproduce_exactly": {"supported": all(cell(layer, corpus)["sites"] == behavior_config["sites_per_corpus"] for layer in parent["layers"] for corpus in arrays)},
        "D1_frozen_series_predicts_midpoint_direction": {"supported": all(cell(layer, corpus)["orders"][str(behavior_config["frozen_orders"][str(layer)])]["midpoint_cosine"] >= .95 for layer in parent["layers"] for corpus in arrays)},
        "D2_midpoint_direction_predicts_finite_endpoint": {"supported": all(cell(layer, corpus)["median_explicit_midpoint_endpoint_cosine"] >= .80 for layer in parent["layers"] for corpus in arrays)},
        "D3_frozen_series_predicts_endpoint_support": {"supported": all(cell(layer, corpus)["orders"][str(behavior_config["frozen_orders"][str(layer)])]["local_support_jaccard"] >= .50 for layer in parent["layers"] for corpus in arrays)},
        "D4_layer0_failure_is_truncation": {"supported": all(cell(0, corpus)["orders"]["5"]["local_support_jaccard"] >= cell(0, corpus)["orders"]["3"]["local_support_jaccard"] + .05 for corpus in arrays)},
        "D5_local_output_scoring_beats_magnitude": {"supported": all(cell(layer, corpus)["orders"][str(behavior_config["frozen_orders"][str(layer)])]["local_write_fidelity"] >= cell(layer, corpus)["orders"][str(behavior_config["frozen_orders"][str(layer)])]["magnitude_write_fidelity"] + .03 for layer in parent["layers"] for corpus in arrays)},
        "D6_longer_edits_reduce_behavioral_recovery": {"supported": all(cell(layer, corpus)["chord_behavior_spearman"] <= -.20 for layer in parent["layers"] for corpus in arrays)},
        "D7_semantic_edits_are_geometrically_easier": {"supported": all(semantic_median <= .90 * cell(0, corpus)["median_chord"] for corpus in arrays)},
        "D8_support_overlap_predicts_behavior": {"supported": all(cell(layer, corpus)["support_behavior_spearman"] >= .30 for layer in parent["layers"] for corpus in arrays)},
        "D9_diagnosis_does_not_refit_orders": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "semantic_median_chord": semantic_median, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Why the frozen Taylor behavioral compiler failed", "", "This diagnostic replays the exact 192 random-token substitutions without refitting orders or sites. It separates derivative truncation, midpoint-versus-finite-path error, support recovery, output scoring, and embedding chord length.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Taylor failure diagnosis](bilin12_layer01_taylor_failure_diagnosis.png)", "", f"Semantic-pair median normalized-embedding chord: **{semantic_median:.3f}**.", "", "| Layer | Corpus | Random chord | Frozen midpoint cosine / endpoint cosine / local support | Frozen local/magnitude write fidelity | Chord→behavior rho | Support→behavior rho |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in parent["layers"]:
        frozen = str(behavior_config["frozen_orders"][str(layer)])
        for corpus in arrays:
            row = cell(layer, corpus); order = row["orders"][frozen]
            lines.append(f"| {layer} | {corpus} | {row['median_chord']:.3f} | {order['midpoint_cosine']:.3f}/{row['median_explicit_midpoint_endpoint_cosine']:.3f}/{order['local_support_jaccard']:.3f} | {order['local_write_fidelity']:.3f}/{order['magnitude_write_fidelity']:.3f} | {row['chord_behavior_spearman']:.3f} | {row['support_behavior_spearman']:.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in parent["layers"]:
        for corpus in arrays:
            row = cell(layer, corpus); orders = range(1, red["maximum_order"] + 1)
            axes[0].plot(orders, [row["orders"][str(order)]["midpoint_cosine"] for order in orders], marker="o", label=f"L{layer} {corpus}")
            axes[1].plot(orders, [row["orders"][str(order)]["local_support_jaccard"] for order in orders], marker="o", label=f"L{layer} {corpus}")
    axes[0].set_title("Cosine to explicit midpoint"); axes[1].set_title("Endpoint-local support Jaccard")
    for axis in axes: axis.set_xlabel("Taylor order"); axis.set_ylim(0, 1.02); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
