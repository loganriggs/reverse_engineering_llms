#!/usr/bin/env python3
"""Test whether CE-important residual geometry is new or generic route covariance."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_ce_adjoint_residual_atlas import fit_basis, reconstruction_fidelity, subspace_cosine
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_ce_adjoint_residual_novelty_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_ce_adjoint_residual_novelty_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_CE_ADJOINT_RESIDUAL_NOVELTY_REDTEAM.md"


def project(values, basis):
    values = np.asarray(values, dtype=np.float64); basis = np.asarray(basis, dtype=np.float64)
    return (values @ basis.T) @ basis


def normalized_complement(values, basis):
    residual = np.asarray(values, dtype=np.float64) - project(values, basis)
    norm = np.linalg.norm(residual, axis=1, keepdims=True)
    return residual / np.maximum(norm, 1e-30)


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text())
    arrays = np.load(config["arrays"]); generic = np.load(config["generic_arrays"])
    rows = []; discovery_novel_bases = {}
    for layer in range(18):
        discovery = arrays[f"discovery_residual_routes_L{layer}"]; replication = arrays[f"replication_residual_routes_L{layer}"]
        gradient_a = arrays[f"discovery_gradients_L{layer}"]; gradient_b = arrays[f"replication_gradients_L{layer}"]
        effects_a = arrays[f"discovery_effects_L{layer}"][:, 2]; effects_b = arrays[f"replication_effects_L{layer}"][:, 2]
        ce_basis = fit_basis(discovery, config["generic_rank"])
        generic_basis = fit_basis(generic[f"discovery_routes_L{layer}"], config["generic_rank"])
        ce_fidelity = reconstruction_fidelity(replication, project(replication, ce_basis))
        generic_fidelity = reconstruction_fidelity(replication, project(replication, generic_basis))
        complement_a = normalized_complement(discovery, generic_basis); complement_b = normalized_complement(replication, generic_basis)
        novel_a = fit_basis(complement_a, config["novel_rank"]); novel_b = fit_basis(complement_b, config["novel_rank"])
        discovery_novel_bases[layer] = novel_a
        gradient_cosine_a = np.abs(np.sum(discovery * gradient_a, axis=1))
        gradient_cosine_b = np.abs(np.sum(replication * gradient_b, axis=1))
        rows.append({
            "layer": layer, "ce_basis_fidelity": ce_fidelity, "generic_basis_fidelity": generic_fidelity,
            "ce_advantage": ce_fidelity - generic_fidelity, "ce_generic_subspace_cosine": subspace_cosine(ce_basis, generic_basis),
            "novel_complement_cosine": subspace_cosine(novel_a, novel_b),
            "gradient_cosine_discovery": float(gradient_cosine_a.mean()), "gradient_cosine_replication": float(gradient_cosine_b.mean()),
            "positive_fraction_discovery": float(np.mean(effects_a > 0)), "positive_fraction_replication": float(np.mean(effects_b > 0)),
        })

    rng = np.random.default_rng(config["seed"]); null_means = []
    for _ in range(config["null_replicates"]):
        scores = []
        for layer in range(18):
            replication = arrays[f"replication_residual_routes_L{layer}"]
            shuffled = np.stack([row[rng.permutation(replication.shape[1])] for row in replication])
            generic_basis = fit_basis(generic[f"discovery_routes_L{layer}"], config["generic_rank"])
            complement = normalized_complement(shuffled, generic_basis)
            null_basis = fit_basis(complement, config["novel_rank"])
            scores.append(subspace_cosine(discovery_novel_bases[layer], null_basis))
        null_means.append(float(np.mean(scores)))
    advantages = [row["layer"] for row in rows if row["ce_advantage"] >= .05]
    nongradient = [row["layer"] for row in rows if max(row["gradient_cosine_discovery"], row["gradient_cosine_replication"]) < .70]
    bipolar = [row["layer"] for row in rows if all(.10 <= row[key] <= .90 for key in ("positive_fraction_discovery", "positive_fraction_replication"))]
    observed = float(np.mean([row["novel_complement_cosine"] for row in rows])); null_p95 = float(np.quantile(null_means, .95))
    hypotheses = {
        "N0_ce_selection_adds_predictive_geometry": {"supported": len(advantages) >= 12, "layers": advantages},
        "N1_top_residual_routes_are_not_just_gradients": {"supported": len(nongradient) == 18, "layers": nongradient},
        "N2_both_helpful_and_harmful_poles_are_present": {"supported": len(bipolar) >= 16, "layers": bipolar},
        "N3_novel_complement_recurrence_beats_head_permutation": {"supported": observed > null_p95, "observed": observed, "null_mean": float(np.mean(null_means)), "null_p95": null_p95},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": rows, "null_mean_novel_cosine": null_means}, indent=2, allow_nan=False))
    lines = ["# Novelty red team for the CE-adjoint residual atlas", "", "The CE-selected residual subspace can recur simply because ordinary route covariance recurs. This report compares it with the earlier task-agnostic high-magnitude route basis, then removes that generic rank-four space and tests rank-two complement recurrence against per-edge head permutations.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | CE/generic fidelity | Advantage | CE↔generic basis | Novel complement cosine | Route↔gradient D/R | Positive pole D/R |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['layer']} | {row['ce_basis_fidelity']:.3f}/{row['generic_basis_fidelity']:.3f} | {row['ce_advantage']:+.3f} | {row['ce_generic_subspace_cosine']:.3f} | {row['novel_complement_cosine']:.3f} | {row['gradient_cosine_discovery']:.3f}/{row['gradient_cosine_replication']:.3f} | {row['positive_fraction_discovery']:.3f}/{row['positive_fraction_replication']:.3f} |")
    lines += ["", "## Guardrail", "", "A recurring complement is still a coordinate-level object, not a semantic feature. It earns fresh token decoding only after beating the generic-basis and destroyed-binding explanations. Because edges were selected for large residual first-order CE effect, effect magnitude itself is not an independent validation target."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
