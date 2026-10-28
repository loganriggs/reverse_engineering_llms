#!/usr/bin/env python3
"""Test whether many output adjoints are served by one shared suffix pullback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


CONFIG = Path("configs/bilin18_layer16_output_conditioned_suffix_atlas.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_output_conditioned_suffix_atlas.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_OUTPUT_CONDITIONED_SUFFIX_ATLAS.md")
MODEL_OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_output_conditioned_suffix_atlas.pt")
COMPONENTS = ("ce", "prediction", "target", "margin")


def fidelity(target, prediction):
    return 1.0 - float(
        (target.double() - prediction.double()).square().sum()
        / target.double().square().sum().clamp_min(1e-30))


@torch.inference_mode()
def pca_basis(rows, maximum_rank):
    values = rows.float().cuda()
    covariance = values.T @ values
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order[:maximum_rank]].cpu()


def balanced_stack(data, components, indices):
    queries = []
    targets = []
    scales = {}
    for component in components:
        q = data["components"][component]["output_covector"][indices].double()
        g = data["components"][component]["write_adjoint"][indices].double()
        scale = (len(g) / g.square().sum().clamp_min(1e-30)).sqrt()
        scales[component] = float(scale)
        queries.append(q * scale)
        targets.append(g * scale)
    return torch.cat(queries), torch.cat(targets), scales


def ridge_fit(query, target, basis, multiplier):
    basis64 = basis.double()
    coefficients = query @ basis64
    gram = coefficients.T @ coefficients
    scale = torch.trace(gram) / len(gram)
    regularized = gram + multiplier * scale * torch.eye(len(gram), dtype=torch.float64)
    weights = torch.linalg.solve(regularized, coefficients.T @ target)
    return {"basis": basis.float(), "weights": weights.float()}


def map_prediction(query, model):
    return (query.float() @ model["basis"]) @ model["weights"]


def write_fidelity(data, component, indices, model):
    query = data["components"][component]["output_covector"][indices]
    target = data["components"][component]["write_adjoint"][indices]
    return fidelity(target, map_prediction(query, model))


@torch.inference_mode()
def behavior_fidelity(data, component, write_prediction, weights):
    state = data["state"].float().cuda()
    forward = data["forward"].float().cuda()
    g = write_prediction.float().cuda()
    left_weight = weights["left"].float().cuda()
    right_weight = weights["right"].float().cuda()
    down = weights["down"].float().cuda()
    left = state @ left_weight.T
    right = state @ right_weight.T
    channel_adjoint = g @ down
    effective = (
        (channel_adjoint * right) @ left_weight
        + (channel_adjoint * left) @ right_weight)
    prediction = torch.einsum("bnd,bd->bn", forward, effective).cpu()
    return fidelity(data["components"][component]["target"], prediction)


def shared_scaled_identity(query, target):
    return float((query * target).sum() / query.square().sum().clamp_min(1e-30))


def validation_average(data, indices, model, components):
    values = [write_fidelity(data, component, indices, model) for component in components]
    return float(np.mean(values)), {component: value for component, value in zip(components, values)}


def factor_recomposition_error(data):
    errors = []
    for key in ("output_covector", "write_adjoint", "effective_adjoint", "target"):
        ce = data["components"]["ce"][key].double()
        difference = (
            data["components"]["prediction"][key].double()
            - data["components"]["target"][key].double())
        errors.append(float((ce - difference).norm() / ce.norm().clamp_min(1e-30)))
    return max(errors)


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    discovery = factors["discovery"]
    evaluations = factors["evaluation"]
    weights = factors["layer16_weights"]
    decomposition = json.loads(Path(config["decomposition_result_path"]).read_text())

    train_end = config["train_sites"]
    validation_end = train_end + config["validation_sites"]
    assert validation_end == len(discovery["state"])
    train_indices = slice(0, train_end)
    validation_indices = slice(train_end, validation_end)
    fit_components = config["fit_components"]

    train_query, train_target, component_scales = balanced_stack(
        discovery, fit_components, train_indices)
    maximum_rank = max(config["subspace_ranks"])
    shared_basis = pca_basis(train_query, maximum_rank)
    shared_grid = []
    selected_model = None
    selected_row = None
    for rank in config["subspace_ranks"]:
        for multiplier in config["ridge_multipliers"]:
            model = ridge_fit(
                train_query, train_target, shared_basis[:, :rank], multiplier)
            value, by_component = validation_average(
                discovery, validation_indices, model, fit_components)
            row = {
                "rank": rank,
                "ridge_multiplier": multiplier,
                "mean_validation_write_fidelity": value,
                "validation_write_fidelity": by_component,
            }
            shared_grid.append(row)
            if selected_row is None or value > selected_row["mean_validation_write_fidelity"]:
                selected_row = row
                selected_model = model

    component_models = {}
    component_selection = {}
    for component in COMPONENTS:
        query = discovery["components"][component]["output_covector"][train_indices].double()
        target = discovery["components"][component]["write_adjoint"][train_indices].double()
        basis = pca_basis(query, maximum_rank)
        best_model = None
        best_row = None
        for rank in config["subspace_ranks"]:
            for multiplier in config["ridge_multipliers"]:
                model = ridge_fit(query, target, basis[:, :rank], multiplier)
                value = write_fidelity(discovery, component, validation_indices, model)
                row = {"rank": rank, "ridge_multiplier": multiplier,
                       "validation_write_fidelity": value}
                if best_row is None or value > best_row["validation_write_fidelity"]:
                    best_row = row
                    best_model = model
        component_models[component] = best_model
        component_selection[component] = best_row

    identity_scale = shared_scaled_identity(train_query, train_target)
    validation_results = {}
    for component in COMPONENTS:
        q = discovery["components"][component]["output_covector"][validation_indices]
        validation_results[component] = {
            "shared_write_fidelity": write_fidelity(
                discovery, component, validation_indices, selected_model),
            "component_write_fidelity": write_fidelity(
                discovery, component, validation_indices, component_models[component]),
            "identity_write_fidelity": fidelity(
                discovery["components"][component]["write_adjoint"][validation_indices],
                identity_scale * q),
            "shared_behavior_fidelity": behavior_fidelity(
                {
                    "state": discovery["state"][validation_indices],
                    "forward": discovery["forward"][validation_indices],
                    "components": {
                        component: {
                            "target": discovery["components"][component]["target"][validation_indices]
                        }
                    },
                }, component, map_prediction(q, selected_model), weights),
        }

    results = {}
    for corpus, data in evaluations.items():
        rows = {}
        for component in COMPONENTS:
            query = data["components"][component]["output_covector"]
            shared_prediction = map_prediction(query, selected_model)
            component_prediction = map_prediction(query, component_models[component])
            identity_prediction = identity_scale * query
            rows[component] = {
                "shared_write_fidelity": fidelity(
                    data["components"][component]["write_adjoint"], shared_prediction),
                "component_write_fidelity": fidelity(
                    data["components"][component]["write_adjoint"], component_prediction),
                "identity_write_fidelity": fidelity(
                    data["components"][component]["write_adjoint"], identity_prediction),
                "shared_behavior_fidelity": behavior_fidelity(
                    data, component, shared_prediction, weights),
                "component_behavior_fidelity": behavior_fidelity(
                    data, component, component_prediction, weights),
                "identity_behavior_fidelity": behavior_fidelity(
                    data, component, identity_prediction, weights),
            }
        results[corpus] = {"sites": len(data["state"]), "components": rows}

    maximum_recomposition = max(
        factor_recomposition_error(discovery),
        *(factor_recomposition_error(data) for data in evaluations.values()),
    )
    spectra = decomposition["discovery_spectra"]["components"]
    tested_components = ("prediction", "target", "margin")
    hypotheses = {
        "O0_saved_factor_recomposition_is_exact": maximum_recomposition <= 1e-6,
        "O1_shared_static_suffix_predicts_write_adjoints": all(
            row["components"][component]["shared_write_fidelity"] >= .80
            for row in results.values() for component in tested_components),
        "O2_shared_static_suffix_predicts_behavior": all(
            row["components"][component]["shared_behavior_fidelity"] >= .80
            for row in results.values() for component in tested_components),
        "O3_shared_static_suffix_explains_model_margin": all(
            row["components"]["margin"]["shared_behavior_fidelity"] >= .90
            for row in results.values()),
        "O4_shared_map_matches_component_specific_ceiling": all(
            row["components"][component]["shared_behavior_fidelity"]
            >= row["components"][component]["component_behavior_fidelity"] - .05
            for row in results.values() for component in tested_components),
        "O5_weighted_map_beats_scaled_identity": all(
            row["components"][component]["shared_behavior_fidelity"]
            >= row["components"][component]["identity_behavior_fidelity"] + .20
            for row in results.values() for component in tested_components),
        "O6_static_suffix_description_is_rank128_or_smaller": selected_row["rank"] <= 128,
        "O7_shared_suffix_map_transfers_ood": all(
            results[name]["components"][component]["shared_behavior_fidelity"]
            >= results["fineweb"]["components"][component]["shared_behavior_fidelity"] - .10
            for name in ("wikitext", "tinystories") for component in tested_components),
        "O8_output_covector_rank_accounts_for_write_rank": all(
            abs(
                spectra[component]["output_covector"]["rank90"]
                - spectra[component]["write_adjoint"]["rank90"]
            ) <= 32 for component in tested_components),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["O0_saved_factor_recomposition_is_exact"]["maximum_relative_error"] = maximum_recomposition
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "component_balance_scales": component_scales,
        "identity_scale": identity_scale,
        "shared_selection": selected_row,
        "shared_validation_grid": shared_grid,
        "component_selection": component_selection,
        "validation": validation_results,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save({
        "shared_model": selected_model,
        "component_models": component_models,
        "identity_scale": identity_scale,
    }, MODEL_OUTPUT)

    lines = [
        "# Output-conditioned suffix atlas at the Layer-16 boundary",
        "",
        "A single linear pullback from final output covectors to Layer-16 write "
        "adjoints is fit jointly across probability, target-logit, and model-margin "
        "queries. Output rank and ridge strength are selected only on a disjoint "
        "FineWeb validation slice. The frozen map is evaluated both at the adjoint "
        "and after exact contraction through the Layer-16 bilinear Jacobian and "
        "natural upstream tangents.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Selected shared rank `{selected_row['rank']}` and ridge multiplier "
        f"`{selected_row['ridge_multiplier']:g}`; mean validation write fidelity "
        f"`{selected_row['mean_validation_write_fidelity']:.3f}`. Saved-factor "
        f"recomposition error: `{maximum_recomposition:.2e}`.",
        "",
        "| Corpus | Component | Shared write | Component-specific write | Identity write | Shared behavior | Component-specific behavior | Identity behavior |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for component in COMPONENTS:
            item = row["components"][component]
            lines.append(
                f"| {corpus} | {component} | {item['shared_write_fidelity']:.3f} | "
                f"{item['component_write_fidelity']:.3f} | {item['identity_write_fidelity']:.3f} | "
                f"{item['shared_behavior_fidelity']:.3f} | "
                f"{item['component_behavior_fidelity']:.3f} | "
                f"{item['identity_behavior_fidelity']:.3f} |")
    lines += [
        "",
        "A successful shared map would mean broad adjoint samples are mostly many "
        "output covectors passed through one stable suffix operator. Failure after "
        "validation selection instead favors context-conditioned pullbacks or a "
        "nonlinear/routed atlas. Component-specific ridge fits are an overpowered "
        "statistical ceiling, not weight-derived circuits.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
