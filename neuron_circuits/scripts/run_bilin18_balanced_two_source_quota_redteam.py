#!/usr/bin/env python3
"""Test whether both positions are causally necessary in balanced two-source edits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_layer1_cross_position_role_redteam import cross_gradient, native_logits
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layers0_3_two_source_selector import logits_for_position_deltas
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_balanced_two_source_quota_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_balanced_two_source_quota_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BALANCED_TWO_SOURCE_QUOTA_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_balanced_two_source_quota_redteam.png"


def keep(vector, indices):
    result = torch.zeros_like(vector); result[indices] = vector[indices]; return result


def summarize(records, layer, pair, method):
    rows = [row for row in records if row["layer"] == layer and row["pair"] == pair]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    global_shifts = np.asarray([row["finite_shifts"]["global64"] for row in rows])
    return {"positive_fraction": float(np.mean(shifts > 0)), "mean_shift": float(shifts.mean()), "global_effect_ratio": float(shifts.mean() / max(global_shifts.mean(), 1e-30))}


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_results"]).read_text())
    source_config = source["config"]; corpus = np.load(config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    sites = [row for row in source["records"] if row["layer"] == config["source_layers"][0]]
    records = []
    for site_index, site in enumerate(sites):
        positions = tuple(site["positions"]); replacements = site["replacement_tokens"]
        raw = np.asarray(corpus[source_config["start_row"] + site["row"], :source_config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        for position, token in zip(positions, replacements): changed[position] = token
        base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_trace = forward_layers_native(model, base_ids, len(model.blocks) - 1)
            changed_trace = forward_layers_native(model, changed_ids, max(config["source_layers"]))
            base_logits = native_logits(model, base_trace, config["prediction_position"])
            changed_logits = native_logits(model, forward_layers_native(model, changed_ids, len(model.blocks) - 1), config["prediction_position"])
        contrast = (F.softmax(changed_logits.float(), -1) - F.softmax(base_logits.float(), -1)).detach()
        for layer in config["source_layers"]:
            width = model.blocks[layer].mlp.Left.out_features
            deltas = [product_at(model, changed_trace, layer, p) - product_at(model, base_trace, layer, p) for p in positions]
            gradients = [cross_gradient(model, base_trace, layer, p, config["prediction_position"], contrast) for p in positions]
            delta = torch.cat(deltas); gradient = torch.cat(gradients); contribution = delta * gradient
            global_support = contribution.topk(config["global_rank"]).indices
            quota_support = torch.cat([contribution[index * width:(index + 1) * width].topk(config["quota_per_position"]).indices + index * width for index in range(2)])
            position_supports = [contribution[index * width:(index + 1) * width].topk(config["global_rank"]).indices + index * width for index in range(2)]
            vectors = {"base": torch.zeros_like(delta), "global64": keep(delta, global_support), "quota32_32": keep(delta, quota_support), "position0_64": keep(delta, position_supports[0]), "position1_64": keep(delta, position_supports[1]), "full_position0": torch.cat([deltas[0], torch.zeros_like(deltas[1])]), "full_position1": torch.cat([torch.zeros_like(deltas[0]), deltas[1]]), "full_both": delta}
            with torch.no_grad(): logits = logits_for_position_deltas(model, layer, base_trace[layer], vectors, positions, config["prediction_position"])
            base_score = float(logits["base"].float() @ contrast)
            finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
            counts = [int(((global_support >= index * width) & (global_support < (index + 1) * width)).sum()) for index in range(2)]
            stronger_support = int(counts[1] > counts[0]); stronger_effect = int(finite["position1_64"] > finite["position0_64"])
            records.append({"pair": site["pair"], "row": site["row"], "layer": layer, "positions": list(positions), "global_support_counts": counts, "support_matches_stronger_effect": stronger_support == stronger_effect, "finite_shifts": finite})
        print(f"quota redteam {site_index + 1}/{len(sites)} {site['pair']}", flush=True)
    pairs = sorted({row["pair"] for row in records}); methods = list(records[0]["finite_shifts"])
    summaries = {str(layer): {pair: {method: summarize(records, layer, pair, method) for method in methods} for pair in pairs} for layer in config["source_layers"]}
    cells = [summaries[str(layer)][pair] for layer in config["source_layers"] for pair in pairs]
    pair_best_single = {}
    for pair in pairs:
        global_mean = np.mean([row["finite_shifts"]["global64"] for row in records if row["pair"] == pair])
        best = max(np.mean([row["finite_shifts"][method] for row in records if row["pair"] == pair]) for method in ("position0_64", "position1_64"))
        pair_best_single[pair] = float(best / max(global_mean, 1e-30))
    match_fraction = float(np.mean([row["support_matches_stronger_effect"] for row in records]))
    hypotheses = {
        "Q0_global_selector_is_causal": {"supported": all(cell["global64"]["positive_fraction"] >= .95 for cell in cells)},
        "Q1_equal_quota_is_efficient": {"supported": all(cell["quota32_32"]["global_effect_ratio"] >= .80 for cell in cells)},
        "Q2_both_sources_are_causally_needed": {"supported": all(value < .90 for value in pair_best_single.values())},
        "Q3_support_allocation_tracks_causal_strength": {"supported": match_fraction >= .80},
        "Q4_far_source_survives_balancing": {"supported": all(summaries[str(layer)]["63+119"]["position0_64"]["global_effect_ratio"] >= .20 for layer in config["source_layers"])},
    }
    result = {"config": config, "hypotheses": hypotheses, "support_strength_match_fraction": match_fraction, "pair_best_single_effect_ratio": pair_best_single, "summaries": summaries, "records": records}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Quota red team of balanced two-source coalitions", "", "The strength-balanced edits are replayed with four causal allocations: an unconstrained global top 64, a forced 32+32 split, and top 64 restricted to either source position. All rankings use the exact local contribution `activation change × suffix gradient`, followed by full nonlinear suffix recomputation.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Support-majority position matches the stronger single-position finite intervention on **{match_fraction:.1%}** of sites.", f"Best single-position/global mean-effect ratios by pair: `{pair_best_single}`.", "", "![Quota red team](bilin18_balanced_two_source_quota_redteam.png)", "", "| Layer | Pair | Quota/global | Position 0/global | Position 1/global |", "|---:|---|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for pair in pairs:
            cell = summaries[str(layer)][pair]
            lines.append(f"| {layer} | {pair} | {cell['quota32_32']['global_effect_ratio']:.3f} | {cell['position0_64']['global_effect_ratio']:.3f} | {cell['position1_64']['global_effect_ratio']:.3f} |")
    lines += ["", "## Guardrail", "", "Equal standalone output-effect norms do not imply equal causal leverage after choosing a source layer. If one position still dominates, that is a routing fact rather than failed balancing. The single-position interventions test necessity for this sparse local payload; they do not remove the corresponding input token or every route by which it influenced the recipient state."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained", sharey=True)
    for axis, pair in zip(axes, pairs):
        for method in ("quota32_32", "position0_64", "position1_64"):
            axis.plot(x, [summaries[str(layer)][pair][method]["global_effect_ratio"] for layer in config["source_layers"]], marker="o", label=method)
        axis.set_xticks(x, config["source_layers"]); axis.set(title=pair, xlabel="Source layer"); axis.grid(alpha=.2); axis.axhline(1, color="gray", linestyle=":")
    axes[0].set_ylabel("Effect / unconstrained top-64"); axes[-1].legend(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "support_strength_match_fraction": match_fraction, "pair_best_single_effect_ratio": pair_best_single}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
