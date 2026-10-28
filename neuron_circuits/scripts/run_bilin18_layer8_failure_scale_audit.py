#!/usr/bin/env python3
"""Disentangle response reconstruction, support selection, and finite nonlinearity."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_suffix_attention_subset import (
    greedy_from_gram,
    prepare_site,
    tensor_program,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_failure_scale_audit.json")
OUTPUT = TENSOR_ROOT / "layer8_failure_scale_audit.json"
REPORT = TENSOR_ROOT / "LAYER8_FAILURE_SCALE_AUDIT.md"


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "q10": float(np.quantile(values, .1)),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def weighted_fidelity(target, predicted):
    energy = target.square().sum().clamp_min(1e-30)
    fidelity = 1 - (predicted - target).square().sum() / energy
    cosine = (target * predicted).sum() / (
        target.norm() * predicted.norm()).clamp_min(1e-30)
    return float(fidelity), float(cosine)


def response(model, site, position, mask):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
    )
    return logits, tangents, output_fisher_response(logits, tangents)


def choose_controls(site_record, failures, pass_threshold):
    candidates = [row for row in site_record["triples"]
                  if row["finite_effect_fidelity"] >= pass_threshold]
    available = set(range(len(candidates)))
    controls = []
    for failure in sorted(failures, key=lambda row: row["whitened_gram_error"]):
        chosen = min(
            available,
            key=lambda index: abs(candidates[index]["whitened_gram_error"]
                                  - failure["whitened_gram_error"]),
        )
        available.remove(chosen)
        controls.append(candidates[chosen])
    # Restore one-to-one alignment with the artifact's failure order.
    lookup = {tuple(row["mask"]): control for row, control in zip(
        sorted(failures, key=lambda row: row["whitened_gram_error"]), controls
    )}
    return [lookup[tuple(row["mask"])] for row in failures]


@torch.inference_mode()
def audit_mask(model, site, layers, position, budget, mask, scales):
    native_logits, _, empty = response(model, site, position, ())
    single = {layer: response(model, site, position, (layer,))[2] for layer in mask}
    pair = {pair_mask: response(model, site, position, pair_mask)[2]
            for pair_mask in itertools.combinations(mask, 2)}
    exact_logits, exact_tangents, exact = response(model, site, position, mask)
    if float((exact_logits - native_logits).abs().max()) > 1e-6:
        raise AssertionError("attention mask changed the native logits")
    # The mask defines the explanatory route used for program selection.  The
    # finite intervention, however, runs the native model with *all* suffix
    # attention paths.  Use the full analytic Jacobian for causal/tangent
    # comparisons rather than mistaking the route-masked tangent for the model's
    # complete local derivative.
    full_logits, full_tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=True,
    )
    if float((full_logits - native_logits).abs().max()) > 1e-6:
        raise AssertionError("full tangent pass changed the native logits")
    full_response = output_fisher_response(full_logits, full_tangents)
    i, j, k = mask
    reconstructed = (pair[(i, j)] + pair[(i, k)] + pair[(j, k)]
                     - single[i] - single[j] - single[k] + empty)
    exact_indices = greedy_from_gram(exact.T @ exact, site["amplitudes"], budget)
    reconstructed_indices = greedy_from_gram(
        reconstructed.T @ reconstructed, site["amplitudes"], budget,
    )

    def normalized_program(matrix, indices):
        denominator = site["amplitudes"][indices].norm().clamp_min(1e-30)
        return matrix[:, indices].sum(-1) / denominator

    fixed = {}
    for name, indices in (("exact", exact_indices),
                          ("reconstructed", reconstructed_indices)):
        fixed[name] = dict(zip(
            ("fidelity", "cosine"),
            weighted_fidelity(
                normalized_program(exact, indices),
                normalized_program(reconstructed, indices),
            ),
        ))
    support_target = normalized_program(full_response, exact_indices)
    support_predicted = normalized_program(full_response, reconstructed_indices)
    support_fidelity, support_cosine = weighted_fidelity(
        support_target, support_predicted,
    )

    scale_rows = {}
    for scale in scales:
        routes = {"exact": exact_indices, "reconstructed": reconstructed_indices}
        native, finite = finite_routes(
            model, site["pre"], site["normalized"], site["x0"], site["value_bus"],
            position, site["mode_directions"], routes, site["output_direction"], scale,
        )
        route_fidelity, route_cosine, target_energy = fisher_fidelity(
            native, finite["exact"]["effect"], finite["reconstructed"]["effect"],
        )
        own = {}
        for name, indices in routes.items():
            raw_norm = site["amplitudes"][indices].norm().clamp_min(1e-30)
            multiplier = scale * site["x"].norm() / raw_norm
            tangent_effect = full_tangents[indices].sum(0) * multiplier
            fidelity, cosine, energy = fisher_fidelity(
                native, finite[name]["effect"], tangent_effect,
            )
            own[name] = {
                "tangent_fidelity": fidelity,
                "tangent_cosine": cosine,
                "finite_effect_energy": energy,
                "native_kl": finite[name]["native_kl"],
            }
        scale_rows[str(scale)] = {
            "between_program_fidelity": route_fidelity,
            "between_program_cosine": route_cosine,
            "target_effect_energy": target_energy,
            "own_tangent": own,
        }
    return {
        "mask": list(mask),
        "response_relative_error": float((exact - reconstructed).norm()
                                         / exact.norm().clamp_min(1e-30)),
        "support_recall": len(set(exact_indices.tolist()) &
                              set(reconstructed_indices.tolist())) / budget,
        "exact_indices": exact_indices.cpu().tolist(),
        "reconstructed_indices": reconstructed_indices.cpu().tolist(),
        "fixed_support_reconstruction": fixed,
        "infinitesimal_between_program_fidelity": support_fidelity,
        "infinitesimal_between_program_cosine": support_cosine,
        "scales": scale_rows,
    }


def aggregate(records, scales):
    result = {}
    for kind in ("failure", "control"):
        rows = [record[kind] for record in records]
        result[kind] = {
            "fixed_exact_fidelity": summary([
                row["fixed_support_reconstruction"]["exact"]["fidelity"] for row in rows
            ]),
            "fixed_reconstructed_fidelity": summary([
                row["fixed_support_reconstruction"]["reconstructed"]["fidelity"]
                for row in rows
            ]),
            "infinitesimal_between_program_fidelity": summary([
                row["infinitesimal_between_program_fidelity"] for row in rows
            ]),
            "support_recall": summary([row["support_recall"] for row in rows]),
            "scales": {},
        }
        for scale in scales:
            key = str(scale)
            result[kind]["scales"][key] = {
                "between_program_fidelity": summary([
                    row["scales"][key]["between_program_fidelity"] for row in rows
                ]),
                "exact_own_tangent_fidelity": summary([
                    row["scales"][key]["own_tangent"]["exact"]["tangent_fidelity"]
                    for row in rows
                ]),
                "reconstructed_own_tangent_fidelity": summary([
                    row["scales"][key]["own_tangent"]["reconstructed"]["tangent_fidelity"]
                    for row in rows
                ]),
            }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    blind = json.loads(Path(config["parent_results"]).read_text())
    blind_config = json.loads(Path(config["parent_config"]).read_text())
    mobius_config = json.loads(Path(blind_config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"], "budget": parent["budget"]}
    layers = parent["suffix_attention_layers"]

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(
        model, source_config, parent["queries"],
    )
    records = []
    for corpus, corpus_result in blind["results"].items():
        corpus_config = blind_config["evaluation"][corpus]
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        for site_record in corpus_result["records"]:
            failures = [row for row in site_record["triples"]
                        if row["finite_effect_fidelity"] < config["failure_fidelity"]]
            if not failures:
                continue
            controls = choose_controls(site_record, failures, config["control_fidelity"])
            raw = rows[site_record["row"]]
            site = prepare_site(
                model, raw, site_record["query"], source_config,
                tensor_basis, directions, vectors,
            )
            for failure, control in zip(failures, controls):
                print(corpus, site_record["query"], failure["mask"], flush=True)
                records.append({
                    "corpus": corpus,
                    "query": site_record["query"],
                    "row": site_record["row"],
                    "original_failure_fidelity": failure["finite_effect_fidelity"],
                    "original_failure_whitened_gram_error": failure["whitened_gram_error"],
                    "original_control_fidelity": control["finite_effect_fidelity"],
                    "original_control_whitened_gram_error": control["whitened_gram_error"],
                    "failure": audit_mask(
                        model, site, layers, source_config["causal_position"],
                        parent["budget"], tuple(failure["mask"]),
                        config["edit_norm_fractions"],
                    ),
                    "control": audit_mask(
                        model, site, layers, source_config["causal_position"],
                        parent["budget"], tuple(control["mask"]),
                        config["edit_norm_fractions"],
                    ),
                })
    aggregated = aggregate(records, config["edit_norm_fractions"])
    failures = aggregated["failure"]
    controls = aggregated["control"]
    smallest = str(config["edit_norm_fractions"][0])
    largest = str(config["edit_norm_fractions"][-1])
    fixed_values = [
        failures["fixed_exact_fidelity"],
        failures["fixed_reconstructed_fidelity"],
    ]
    own_small = [
        failures["scales"][smallest]["exact_own_tangent_fidelity"],
        failures["scales"][smallest]["reconstructed_own_tangent_fidelity"],
    ]
    hypotheses = {
        "S0_fixed_support_reconstruction_is_accurate": {
            "supported": all(row["median"] >= .99 and row["min"] >= .90
                             for row in fixed_values),
        },
        "S1_selected_programs_disagree_in_the_linear_limit": {
            "supported": failures["infinitesimal_between_program_fidelity"]["median"] < .80,
        },
        "S2_finite_nonlinearity_causes_the_failure": {
            "supported": (failures["scales"][smallest]["between_program_fidelity"]["median"] >= .80
                          and failures["scales"][smallest]["between_program_fidelity"]["median"]
                          >= failures["scales"][largest]["between_program_fidelity"]["median"] + .20),
        },
        "S3_each_program_is_locally_predicted_by_its_tangent": {
            "supported": all(row["median"] >= .95 and row["q10"] >= .80
                             for row in own_small),
        },
        "S4_failures_are_specific_to_unstable_program_selection": {
            "supported": (controls["infinitesimal_between_program_fidelity"]["median"] >= .95
                          and controls["scales"][largest]["between_program_fidelity"]["median"] >= .95
                          and failures["infinitesimal_between_program_fidelity"]["median"] < .80
                          and failures["scales"][largest]["between_program_fidelity"]["median"] < .80),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses,
        "summary": aggregated, "records": records,
    }, indent=2))

    lines = [
        "# Scale and support audit of blind-replication failures", "",
        "This audit separates response reconstruction on fixed supports, discontinuous sparse-program selection, and finite nonlinear propagation for the twelve blind failures and same-site error-matched passing controls.",
        "", "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "", "## Fixed-support and support-selection results", "",
        "| Group | Fixed exact support fidelity (median/min) | Fixed reconstructed support fidelity (median/min) | Exact vs reconstructed selected programs, infinitesimal (median/q10) | Support recall (median) |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("failure", "control"):
        row = aggregated[kind]
        lines.append(
            f"| {kind} | {row['fixed_exact_fidelity']['median']:.3f}/{row['fixed_exact_fidelity']['min']:.3f} | "
            f"{row['fixed_reconstructed_fidelity']['median']:.3f}/{row['fixed_reconstructed_fidelity']['min']:.3f} | "
            f"{row['infinitesimal_between_program_fidelity']['median']:.3f}/{row['infinitesimal_between_program_fidelity']['q10']:.3f} | "
            f"{row['support_recall']['median']:.3f} |"
        )
    lines += [
        "", "## Scale sweep", "",
        "| Group | Edit norm | Between-program fidelity median/q10 | Exact support: own-tangent fidelity median/q10 | Reconstructed support: own-tangent fidelity median/q10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("failure", "control"):
        for scale in config["edit_norm_fractions"]:
            row = aggregated[kind]["scales"][str(scale)]
            lines.append(
                f"| {kind} | {scale:.4f} | "
                f"{row['between_program_fidelity']['median']:.3f}/{row['between_program_fidelity']['q10']:.3f} | "
                f"{row['exact_own_tangent_fidelity']['median']:.3f}/{row['exact_own_tangent_fidelity']['q10']:.3f} | "
                f"{row['reconstructed_own_tangent_fidelity']['median']:.3f}/{row['reconstructed_own_tangent_fidelity']['q10']:.3f} |"
            )
    lines += [
        "",
        "A failure here is not automatically a model-computation failure: independently maximizing a non-convex/discrete sparse-program objective can make the explanatory program non-identifiable even when the response tensor is accurately reconstructed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
