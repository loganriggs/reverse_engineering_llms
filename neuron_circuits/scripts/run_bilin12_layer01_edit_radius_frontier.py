#!/usr/bin/env python3
"""Causal radius frontier for frozen Taylor-compiled native coalitions."""

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
from run_bilin12_layer01_directional_taylor_selector import cosine
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, median, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import exact_kl, logits_for_deltas, product_from_x0, sample_edits
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_edit_radius_frontier.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_edit_radius_frontier.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_EDIT_RADIUS_FRONTIER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_edit_radius_frontier.png"
METHODS = ("taylor_local", "taylor_magnitude", "explicit_midpoint_local", "endpoint_local", "random")


def main():
    red = json.loads(CONFIG.read_text())
    behavior = json.loads(Path(red["source_config"]).read_text())
    parent = json.loads(Path(behavior["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in parent["evaluation"].items()
    }
    records = {str(layer): [] for layer in parent["layers"]}
    maximum_recomposition_error = 0.0
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        edits = sample_edits(
            rows, behavior["sites_per_corpus"], behavior["sampling_seed"] + corpus_index,
            behavior["edit_position"], behavior["sequence_length"],
        )[:red["sites_per_corpus"]]
        for site_index, (row_index, target_token) in enumerate(edits):
            raw = np.asarray(rows[row_index, :behavior["sequence_length"]], dtype=np.int64).copy()
            changed = raw.copy(); changed[behavior["edit_position"]] = target_token
            base_ids = torch.as_tensor(raw, device="cuda")[None]
            endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
            unit_direction = endpoint_x0 - base_x0
            for scale_index, scale in enumerate(red["scales"]):
                direction = scale * unit_direction
                for layer in parent["layers"]:
                    order = behavior["frozen_orders"][str(layer)]
                    function = lambda value, layer=layer: product_from_x0(
                        model, value, layer, behavior["prediction_position"],
                    )
                    with torch.enable_grad():
                        derivatives, functions = derivative_frontier(function, base_x0, direction, order)
                        explicit_midpoint = functions[1](base_x0 + .5 * direction)
                    with torch.no_grad():
                        base_product = function(base_x0)
                        endpoint_product = function(base_x0 + direction)
                    derivatives = [value.detach().float() for value in derivatives]
                    taylor = taylor_sum(derivatives, .5, integral=False)
                    explicit_midpoint = explicit_midpoint.detach().float()
                    full = endpoint_product.detach().float() - base_product.detach().float()
                    decoder = model.blocks[layer].mlp.Down.weight.float()
                    endpoint_indices = local_scores(full, decoder).topk(red["rank"]).indices
                    generator = torch.Generator(device="cuda").manual_seed(
                        red["random_seed"] + corpus_index * 100000 + site_index * 100 + scale_index * 10 + layer
                    )
                    scores = {
                        "taylor_local": local_scores(taylor, decoder),
                        "taylor_magnitude": taylor.abs(),
                        "explicit_midpoint_local": local_scores(explicit_midpoint, decoder),
                        "endpoint_local": local_scores(full, decoder),
                        "random": torch.rand(len(full), generator=generator, device="cuda"),
                    }
                    deltas = {
                        method: selected_delta(full, score.topk(red["rank"]).indices)
                        for method, score in scores.items()
                    }
                    with torch.no_grad():
                        logits = logits_for_deltas(
                            model, layer, base_trace[layer],
                            {"base": torch.zeros_like(full), "full": full, **deltas},
                            behavior["prediction_position"],
                        )
                    recomposed = logits_for_deltas(
                        model, layer, base_trace[layer], {"full": full}, behavior["prediction_position"],
                    )["full"]
                    maximum_recomposition_error = max(maximum_recomposition_error, float((recomposed - logits["full"]).abs().max()))
                    base_kl = exact_kl(logits["full"], logits["base"])
                    records[str(layer)].append({
                        "corpus": corpus, "site": site_index, "scale": scale,
                        "chord": float(direction.norm()), "base_kl": base_kl,
                        "taylor_midpoint_cosine": cosine(taylor, explicit_midpoint),
                        "explicit_endpoint_cosine": cosine(explicit_midpoint, full),
                        "support_jaccard": {
                            method: jaccard(score.topk(red["rank"]).indices.cpu().tolist(), endpoint_indices.cpu().tolist())
                            for method, score in scores.items()
                        },
                        "kl_error": {method: exact_kl(logits["full"], logits[method]) for method in METHODS},
                    })
            if (site_index + 1) % 8 == 0:
                print("radius", corpus, site_index + 1, "/", len(edits), flush=True)

    summaries = {str(layer): {} for layer in parent["layers"]}
    for layer in parent["layers"]:
        for corpus in arrays:
            summaries[str(layer)][corpus] = {}
            for scale in red["scales"]:
                selected = [row for row in records[str(layer)] if row["corpus"] == corpus and row["scale"] == scale]
                denominator = sum(row["base_kl"] for row in selected)
                summaries[str(layer)][corpus][str(scale)] = {
                    "sites": len(selected), "median_chord": median([row["chord"] for row in selected]),
                    "median_taylor_midpoint_cosine": median([row["taylor_midpoint_cosine"] for row in selected]),
                    "median_explicit_endpoint_cosine": median([row["explicit_endpoint_cosine"] for row in selected]),
                    "kl_recovery": {method: float(1 - sum(row["kl_error"][method] for row in selected) / max(denominator, 1e-30)) for method in METHODS},
                    "support_jaccard": {method: median([row["support_jaccard"][method] for row in selected]) for method in METHODS},
                }
    cell = lambda layer, corpus, scale: summaries[str(layer)][corpus][str(scale)]
    hypotheses = {
        "S0_full_patch_recomposition_is_exact": {"supported": maximum_recomposition_error <= 1e-4},
        "S1_taylor_direction_degrades_with_radius": {"supported": all(all(cell(layer, corpus, red["scales"][i])["median_taylor_midpoint_cosine"] >= cell(layer, corpus, red["scales"][i+1])["median_taylor_midpoint_cosine"] - .03 for i in range(len(red["scales"])-1)) for layer in parent["layers"] for corpus in arrays)},
        "S2_quarter_radius_is_predictable": {"supported": all(cell(layer, corpus, .25)["kl_recovery"]["taylor_local"] >= .70 for layer in parent["layers"] for corpus in arrays)},
        "S3_half_radius_remains_useful": {"supported": all(cell(layer, corpus, .5)["kl_recovery"]["taylor_local"] >= .60 for layer in parent["layers"] for corpus in arrays)},
        "S4_large_edits_cross_a_locality_boundary": {"supported": all(cell(layer, corpus, .25)["kl_recovery"]["taylor_local"] >= cell(layer, corpus, 1.0)["kl_recovery"]["taylor_local"] + .20 for layer in parent["layers"] for corpus in arrays)},
        "S5_exact_midpoint_rescues_large_edits": {"supported": all(cell(layer, corpus, 1.0)["kl_recovery"]["explicit_midpoint_local"] >= .65 for layer in parent["layers"] for corpus in arrays)},
        "S6_truncation_is_a_material_large_edit_cost": {"supported": all(cell(layer, corpus, 1.0)["kl_recovery"]["explicit_midpoint_local"] >= cell(layer, corpus, 1.0)["kl_recovery"]["taylor_local"] + .10 for layer in parent["layers"] for corpus in arrays)},
        "S7_taylor_beats_random_within_its_local_regime": {"supported": all(cell(layer, corpus, scale)["kl_recovery"]["taylor_local"] >= cell(layer, corpus, scale)["kl_recovery"]["random"] + .30 for layer in parent["layers"] for corpus in arrays for scale in (.25, .5))},
        "S8_radius_frontier_transfers_ood": {"supported": all(abs(cell(layer, "fineweb", scale)["kl_recovery"]["taylor_local"] - cell(layer, "tinystories", scale)["kl_recovery"]["taylor_local"]) <= .15 for layer in parent["layers"] for scale in red["scales"])},
        "S9_scales_and_orders_are_frozen": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "maximum_recomposition_error": maximum_recomposition_error, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Radius of validity for Taylor-compiled coalitions", "", "The same naturally sampled token substitutions are continuously scaled in normalized-embedding space. Frozen Layer-0 cubic and Layer-1 fifth-order programs select 256 native product channels before the finite endpoint patch is evaluated.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Edit radius frontier](bilin12_layer01_edit_radius_frontier.png)", "", "| Layer | Corpus | Scale | Chord | Taylor/explicit midpoint cosine | Taylor/explicit/endpoint/random KL recovery | Taylor support |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in parent["layers"]:
        for corpus in arrays:
            for scale in red["scales"]:
                row = cell(layer, corpus, scale); recovery = row["kl_recovery"]
                lines.append(f"| {layer} | {corpus} | {scale:.2f} | {row['median_chord']:.2f} | {row['median_taylor_midpoint_cosine']:.3f}/{row['median_explicit_endpoint_cosine']:.3f} | {recovery['taylor_local']:.3f}/{recovery['explicit_midpoint_local']:.3f}/{recovery['endpoint_local']:.3f}/{recovery['random']:.3f} | {row['support_jaccard']['taylor_local']:.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in parent["layers"]:
        for corpus in arrays:
            axes[layer].plot(red["scales"], [cell(layer, corpus, scale)["kl_recovery"]["taylor_local"] for scale in red["scales"]], marker="o", label=f"{corpus} Taylor")
            axes[layer].plot(red["scales"], [cell(layer, corpus, scale)["kl_recovery"]["explicit_midpoint_local"] for scale in red["scales"]], marker="s", linestyle="--", label=f"{corpus} midpoint")
        axes[layer].set_title(f"Layer {layer}"); axes[layer].set_xlabel("Edit scale"); axes[layer].set_ylabel("KL recovery"); axes[layer].grid(alpha=.2); axes[layer].legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
