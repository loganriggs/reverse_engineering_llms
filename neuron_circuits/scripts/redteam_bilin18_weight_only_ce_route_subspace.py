#!/usr/bin/env python3
"""Test weight-basis reconstruction against dimensional and empirical nulls."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_ce_adjoint_residual_atlas import fit_basis, reconstruction_fidelity
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_weight_only_ce_route_subspace import random_basis


CONFIG = Path("configs/bilin18_weight_only_ce_route_subspace_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_weight_only_ce_route_subspace_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_ONLY_CE_ROUTE_SUBSPACE_REDTEAM.md"


def fidelity(values, basis):
    return reconstruction_fidelity(values, (values @ basis.T) @ basis)


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text())
    bases = np.load(config["weight_bases"]); residual = np.load(config["residual_arrays"]); generic = np.load(config["generic_route_arrays"])
    rng = np.random.default_rng(config["seed"]); rows = []
    for row in source["layers"]:
        layer = row["layer"]; values = residual[f"replication_residual_routes_L{layer}"]
        method = row["best_method"]; weight_basis = bases[f"{method}_basis_L{layer}"]
        weight_fidelity = fidelity(values, weight_basis)
        generic_basis = fit_basis(generic[f"discovery_routes_L{layer}"], config["rank"]); generic_fidelity = fidelity(values, generic_basis)
        null = np.asarray([
            max(fidelity(values, random_basis(values.shape[1], config["rank"], rng)) for _ in range(config["searched_views"]))
            for _ in range(config["null_replicates"])
        ])
        rows.append({
            "layer": layer, "method": method, "weight_fidelity": weight_fidelity, "generic_fidelity": generic_fidelity,
            "weight_minus_generic": weight_fidelity - generic_fidelity,
            "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
            "null_percentile": float(np.mean(null <= weight_fidelity)),
        })
    above_null = [row["layer"] for row in rows if row["weight_fidelity"] > row["null_p95"]]
    above_generic = [row["layer"] for row in rows if row["weight_minus_generic"] >= .02]
    early = [layer for layer in above_generic if layer in config["early_novel_layers"]]
    original_passed_all = len(source["hypotheses"]["W3_weight_basis_recovers_material_ce_residual_geometry"]["layers"]) == 18
    hypotheses = {
        "R0_material_weight_reconstruction_beats_dimensional_null": {"supported": len(above_null) >= 12, "layers": above_null},
        "R1_weight_geometry_beats_empirical_generic_routes": {"supported": len(above_generic) >= 9, "layers": above_generic},
        "R2_early_weight_signal_is_selective": {"supported": len(early) >= 2, "layers": early},
        "R3_half_ce_threshold_was_too_weak": {"supported": len(above_null) < 12 and original_passed_all, "original_passed_all": original_passed_all, "null_passing_layers": len(above_null)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": rows}, indent=2, allow_nan=False))
    lines = ["# Reconstruction red team for weight-only CE-route bases", "", "The original half-of-CE fidelity gate can pass by dimensionality alone: four directions inhabit a nine-dimensional head space. This audit evaluates the discovery-selected weight view against the best of four random rank-four bases on the same replication routes, and against the earlier empirical natural-route basis.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Weight view | Weight fidelity | Best-4 random mean/p95/pct | Generic fidelity | Weight−generic |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['layer']} | {row['method']} | {row['weight_fidelity']:.3f} | {row['null_mean']:.3f}/{row['null_p95']:.3f}/{row['null_percentile']:.3f} | {row['generic_fidelity']:.3f} | {row['weight_minus_generic']:+.3f} |")
    lines += ["", "## Interpretation", "", "A stable static basis can still be a valid description of the checkpoint's operator inventory even when it fails to predict which coordinates FineWeb and next-token loss execute. Passing the empirical-route comparison only in selected early layers would support a hybrid ontology: weights delimit reusable operations, while context/frequency selects their natural coordinate chart."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
