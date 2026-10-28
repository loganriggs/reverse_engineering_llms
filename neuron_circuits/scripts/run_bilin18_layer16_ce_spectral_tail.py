#!/usr/bin/env python3
"""Test whether Layer-16 loss sensitivity resides in the local spectral tail."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from run_bilin18_layer16_ce_aware_mode_selection import (
    collect, contributions, greedy_order, score,
)
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets
from run_bilin18_late_mlp_tensor_canonicalization import exact_reads, gram_modes
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_ce_spectral_tail.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_spectral_tail.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CE_SPECTRAL_TAIL.md"


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    atlas_states = collect_states(model, config["local_atlas_discovery"], config)[16]
    atlas_target = derivative_targets(model.blocks[16], atlas_states, axes, config["batch_size"])
    modes, _, spectrum = gram_modes(atlas_target, config["operator_modes"])
    if len(modes) < config["operator_modes"]:
        raise RuntimeError(f"Only {len(modes)} numerically stable modes; locked budgets require {config['operator_modes']}")
    reads = exact_reads(model.blocks[16], axes, modes)
    print("selection discovery", flush=True)
    discovery = collect(model, config["selection_discovery"], config, axes, validate=True)
    discovery_contribution = contributions(discovery, modes, reads, config["rank"])
    ordering = greedy_order(discovery_contribution, discovery["target"])
    discovery_rows = {str(k): {
        "ce_aware_fidelity": score(discovery["target"], discovery_contribution[:, ordering[:k]].sum(1)),
        "energy_prefix_fidelity": score(discovery["target"], discovery_contribution[:, :k].sum(1))}
        for k in config["budgets"]}
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, config, axes)
        contribution = contributions(data, modes, reads, config["rank"])
        rows = {str(k): {
            "ce_aware_fidelity": score(data["target"], contribution[:, ordering[:k]].sum(1)),
            "energy_prefix_fidelity": score(data["target"], contribution[:, :k].sum(1))}
            for k in config["budgets"]}
        k90 = next((k for k in config["budgets"] if rows[str(k)]["ce_aware_fidelity"] >= .90), None)
        results[corpus] = {"sites": len(data["target"]), "k90": k90, "budgets": rows}
    overlap = len(set(ordering[:256]) & set(range(256)))
    validation_error = discovery["forward_reverse_relative_error"]
    hypotheses = {
        "T0_reverse_ce_derivative_matches_forward_jvp": validation_error <= 1e-5,
        "T1_energy_prefix512_is_sufficient": all(row["budgets"]["512"]["energy_prefix_fidelity"] >= .95 for row in results.values()),
        "T2_near_full_1024_atlas_is_high_fidelity": all(row["budgets"]["1024"]["energy_prefix_fidelity"] >= .99 for row in results.values()),
        "T3_ce_aware_256_is_sufficient": all(row["budgets"]["256"]["ce_aware_fidelity"] >= .90 for row in results.values()),
        "T4_ce_aware_512_is_high_fidelity": all(row["budgets"]["512"]["ce_aware_fidelity"] >= .97 for row in results.values()),
        "T5_loss_k90_is_at_most_256": all(row["k90"] is not None and row["k90"] <= 256 for row in results.values()),
        "T6_tail_program_transfers_ood": all(results[name]["budgets"]["512"]["ce_aware_fidelity"] >= results["fineweb"]["budgets"]["512"]["ce_aware_fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "T7_loss_relevance_strongly_reorders_top256": overlap < 128,
        "T8_spectral_tail_is_behaviorally_material": all(row["budgets"]["1024"]["energy_prefix_fidelity"] >= row["budgets"]["256"]["energy_prefix_fidelity"] + .05 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["T0_reverse_ce_derivative_matches_forward_jvp"]["relative_error"] = validation_error
    hypothesis_rows["T7_loss_relevance_strongly_reorders_top256"]["top256_overlap"] = overlap
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "atlas_discovery_spectrum": spectrum, "ce_aware_mode_order": ordering,
               "selection_discovery_budgets": discovery_rows, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# CE relevance in the near-full Layer-16 spectral tail", "",
             "The local atlas is enlarged from 256 to 1,024 uncentered modes using 1,024 FineWeb position-sites. Every mode keeps its exact weight-derived state coefficient. Fixed CE-aware selection and the ordinary local-energy prefix are evaluated on disjoint 128-site panels.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Forward/reverse error: `{validation_error:.2e}`. CE-aware/energy top-256 overlap: `{overlap}/256`. Atlas discovery energy R90: `{spectrum['rank90']}`.",
              f"Discovery K256 CE-aware/energy: `{discovery_rows['256']['ce_aware_fidelity']:.3f}` / `{discovery_rows['256']['energy_prefix_fidelity']:.3f}`.", "",
              "| Corpus | K | CE-aware | Energy prefix |", "|---|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['ce_aware_fidelity']:.3f} | {item['energy_prefix_fidelity']:.3f} |")
        lines.append(f"| {corpus} | K90 | {row['k90'] if row['k90'] is not None else '>1024'} | — |")
    lines += ["", "If fidelity rises mainly by including the tail rather than by reordering a small bank, the loss computation is genuinely broad in this operator basis. That still does not imply 1,024 semantic mechanisms: modes are dense coordinates, and the endpoint is a scalar training-loss derivative rather than all behaviors."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
