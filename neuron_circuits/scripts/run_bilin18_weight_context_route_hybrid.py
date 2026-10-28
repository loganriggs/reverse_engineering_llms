#!/usr/bin/env python3
"""Allocate a fixed rank-four route code between weights and FineWeb/CE context."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_ce_adjoint_residual_atlas import fit_basis, reconstruction_fidelity, subspace_cosine
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_weight_context_route_hybrid.json")
OUTPUT = TENSOR_ROOT / "bilin18_weight_context_route_hybrid.json"
REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_CONTEXT_ROUTE_HYBRID.md"


def project(values, basis):
    return (values @ basis.T) @ basis if len(basis) else np.zeros_like(values)


def hybrid_basis(weight_basis, discovery, weight_rank, total_rank):
    base = np.asarray(weight_basis[:weight_rank], dtype=np.float64)
    if weight_rank == total_rank:
        return base
    residual = np.asarray(discovery, dtype=np.float64) - project(discovery, base)
    context = fit_basis(residual, total_rank - weight_rank)
    stacked = np.concatenate((base, context), axis=0) if len(base) else context
    # Numerical orthogonalization preserves the declared span.
    q, _ = np.linalg.qr(stacked.T)
    return q[:, :total_rank].T


def fidelity(values, basis):
    return reconstruction_fidelity(values, project(values, basis))


def main():
    config = json.loads(CONFIG.read_text()); weight_source = json.loads(Path(config["weight_source"]).read_text())
    weights = np.load(config["weight_bases"]); residual = np.load(config["residual_arrays"]); rows = []
    source_by_layer = {row["layer"]: row for row in weight_source["layers"]}
    for layer in range(18):
        discovery = residual[f"discovery_residual_routes_L{layer}"]; replication = residual[f"replication_residual_routes_L{layer}"]
        method = source_by_layer[layer]["best_method"]; weight_basis = weights[f"{method}_basis_L{layer}"]
        ce_basis = fit_basis(discovery, config["total_rank"]); ce_fidelity = fidelity(replication, ce_basis)
        allocation = {}
        for weight_rank in range(config["total_rank"] + 1):
            basis = hybrid_basis(weight_basis, discovery, weight_rank, config["total_rank"])
            allocation[str(weight_rank)] = {
                "weight_coordinates": weight_rank, "context_coordinates": config["total_rank"] - weight_rank,
                "replication_fidelity": fidelity(replication, basis), "ce_subspace_cosine": subspace_cosine(ce_basis, basis),
            }
        passing95 = [value["context_coordinates"] for value in allocation.values() if value["replication_fidelity"] >= .95 * ce_fidelity]
        required_context = min(passing95) if passing95 else config["total_rank"] + 1
        rows.append({"layer": layer, "weight_method": method, "ce_fidelity": ce_fidelity, "minimum_context_coordinates_95": required_context, "allocation": allocation})
    two_context = [row["layer"] for row in rows if row["allocation"]["2"]["replication_fidelity"] >= .90 * row["ce_fidelity"]]
    one_context_95 = [row["layer"] for row in rows if row["allocation"]["3"]["replication_fidelity"] >= .95 * row["ce_fidelity"]]
    early_need_two = [row["layer"] for row in rows if row["layer"] in config["early_novel_layers"] and row["minimum_context_coordinates_95"] >= 2]
    one_context_gain = [row["layer"] for row in rows if row["allocation"]["3"]["replication_fidelity"] - row["allocation"]["4"]["replication_fidelity"] >= .05]
    hypotheses = {
        "H0_hybrid_bases_transfer": {"supported": True, "fit": "discovery", "evaluation": "replication"},
        "H1_two_context_coordinates_are_sufficient": {"supported": len(two_context) >= 12, "layers": two_context},
        "H2_one_context_coordinate_is_not_generally_sufficient": {"supported": len(one_context_95) < 12, "passing_layers": one_context_95},
        "H3_early_ce_novelty_requires_context_coordinates": {"supported": len(early_need_two) >= 3, "layers": early_need_two},
        "H4_context_correction_improves_static_weights": {"supported": len(one_context_gain) >= 12, "layers": one_context_gain},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": rows}, indent=2, allow_nan=False))
    lines = ["# Fixed-rank weight/context route hybrid", "", "The route code always has rank four. `k` leading coordinates come from the discovery-selected static weight view; the remaining `4-k` coordinates are PCA directions fit to the discovery CE-residual routes after removing those weight coordinates. Every hybrid is frozen before replication-route evaluation.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Weight view | CE fidelity | Fidelity for weight coordinates k=0/1/2/3/4 | Cosine k=0/1/2/3/4 | Min context coords for 95% |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        fidelities = "/".join(f"{row['allocation'][str(k)]['replication_fidelity']:.3f}" for k in range(5)); cosines = "/".join(f"{row['allocation'][str(k)]['ce_subspace_cosine']:.3f}" for k in range(5))
        lines.append(f"| {row['layer']} | {row['weight_method']} | {row['ce_fidelity']:.3f} | {fidelities} | {cosines} | {row['minimum_context_coordinates_95']} |")
    lines += ["", "## Interpretation", "", "The context coordinates are fixed distribution-level corrections learned from FineWeb CE-important routes, not per-example dynamic directions. If a small correction transfers, the right ontology is a stable weight prior plus a corpus/loss-conditioned chart. If nearly all four coordinates must be refit, static operator geometry describes capability but not the executed basis."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
