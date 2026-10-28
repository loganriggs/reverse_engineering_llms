#!/usr/bin/env python3
"""Red-team the static suffix null with low-rank corrections around identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_output_conditioned_suffix_atlas import (
    COMPONENTS,
    balanced_stack,
    behavior_fidelity,
    factor_recomposition_error,
    fidelity,
    map_prediction,
    pca_basis,
    ridge_fit,
    shared_scaled_identity,
)


CONFIG = Path("configs/bilin18_layer16_suffix_identity_residual_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_suffix_identity_residual_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_SUFFIX_IDENTITY_RESIDUAL_REDTEAM.md")
MODEL_OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_suffix_identity_residual_redteam.pt")


def residual_prediction(query, model):
    return model["identity_scale"] * query.float() + map_prediction(query, model)


def write_fidelity(data, component, indices, model):
    query = data["components"][component]["output_covector"][indices]
    target = data["components"][component]["write_adjoint"][indices]
    return fidelity(target, residual_prediction(query, model))


def fit_residual_model(query, target, basis, multiplier, identity_scale):
    residual = target - identity_scale * query
    model = ridge_fit(query, residual, basis, multiplier)
    model["identity_scale"] = float(identity_scale)
    return model


def validation_average(data, indices, model, components):
    values = [write_fidelity(data, component, indices, model) for component in components]
    return float(np.mean(values)), {component: value for component, value in zip(components, values)}


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    discovery = factors["discovery"]
    evaluations = factors["evaluation"]
    weights = factors["layer16_weights"]
    train_end = config["train_sites"]
    validation_end = train_end + config["validation_sites"]
    assert validation_end == len(discovery["state"])
    train_indices = slice(0, train_end)
    validation_indices = slice(train_end, validation_end)
    fit_components = config["fit_components"]

    train_query, train_target, component_scales = balanced_stack(
        discovery, fit_components, train_indices)
    identity_scale = shared_scaled_identity(train_query, train_target)
    maximum_rank = max(config["subspace_ranks"])
    shared_basis = pca_basis(train_query, maximum_rank)
    shared_grid = []
    selected_model = None
    selected_row = None
    for rank in config["subspace_ranks"]:
        for multiplier in config["ridge_multipliers"]:
            model = fit_residual_model(
                train_query, train_target, shared_basis[:, :rank],
                multiplier, identity_scale)
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
        component_identity = shared_scaled_identity(query, target)
        basis = pca_basis(query, maximum_rank)
        best_model = None
        best_row = None
        for rank in config["subspace_ranks"]:
            for multiplier in config["ridge_multipliers"]:
                model = fit_residual_model(
                    query, target, basis[:, :rank], multiplier, component_identity)
                value = write_fidelity(discovery, component, validation_indices, model)
                row = {"rank": rank, "ridge_multiplier": multiplier,
                       "identity_scale": component_identity,
                       "validation_write_fidelity": value}
                if best_row is None or value > best_row["validation_write_fidelity"]:
                    best_row = row
                    best_model = model
        component_models[component] = best_model
        component_selection[component] = best_row

    baseline_validation = np.mean([
        fidelity(
            discovery["components"][component]["write_adjoint"][validation_indices],
            identity_scale * discovery["components"][component]["output_covector"][validation_indices],
        ) for component in fit_components
    ])
    results = {}
    for corpus, data in evaluations.items():
        rows = {}
        for component in COMPONENTS:
            query = data["components"][component]["output_covector"]
            shared_prediction = residual_prediction(query, selected_model)
            component_prediction = residual_prediction(query, component_models[component])
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
    tested_components = ("prediction", "target", "margin")
    hypotheses = {
        "I0_saved_factor_recomposition_is_exact": maximum_recomposition <= 1e-6,
        "I1_identity_residual_improves_validation_write": (
            selected_row["mean_validation_write_fidelity"] >= baseline_validation + .10),
        "I2_identity_residual_predicts_behavior": all(
            row["components"][component]["shared_behavior_fidelity"] >= .80
            for row in results.values() for component in tested_components),
        "I3_identity_residual_explains_model_margin": all(
            row["components"]["margin"]["shared_behavior_fidelity"] >= .80
            for row in results.values()),
        "I4_shared_residual_matches_component_specific_ceiling": all(
            row["components"][component]["shared_behavior_fidelity"]
            >= row["components"][component]["component_behavior_fidelity"] - .05
            for row in results.values() for component in tested_components),
        "I5_identity_residual_is_rank512_or_smaller": selected_row["rank"] <= 512,
        "I6_identity_residual_transfers_ood": all(
            results[name]["components"][component]["shared_behavior_fidelity"]
            >= results["fineweb"]["components"][component]["shared_behavior_fidelity"] - .10
            for name in ("wikitext", "tinystories") for component in tested_components),
        "I7_identity_residual_avoids_behavioral_sign_failure": all(
            row["components"][component]["shared_behavior_fidelity"] >= 0
            for row in results.values() for component in COMPONENTS),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["I0_saved_factor_recomposition_is_exact"]["maximum_relative_error"] = maximum_recomposition
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "component_balance_scales": component_scales,
        "identity_scale": identity_scale,
        "identity_validation_write_fidelity": baseline_validation,
        "shared_selection": selected_row,
        "shared_validation_grid": shared_grid,
        "component_selection": component_selection,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save({
        "shared_model": selected_model,
        "component_models": component_models,
        "identity_scale": identity_scale,
    }, MODEL_OUTPUT)

    lines = [
        "# Identity-residual red-team for the output-conditioned suffix atlas",
        "",
        "The failed zero-centered suffix regression is replaced by a fitted shared "
        "scaled identity plus a low-rank ridge correction. This preserves the strong "
        "full-rank baseline and tests ranks through 768. Rank and regularization are "
        "again selected only on disjoint FineWeb validation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Scaled-identity validation write fidelity: `{baseline_validation:.3f}`. "
        f"Selected residual rank `{selected_row['rank']}`, ridge multiplier "
        f"`{selected_row['ridge_multiplier']:g}`, validation write fidelity "
        f"`{selected_row['mean_validation_write_fidelity']:.3f}`.",
        "",
        "| Corpus | Component | Shared write | Component write | Identity write | Shared behavior | Component behavior | Identity behavior |",
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
        "If this residual model remains weak, the zero-centered failure cannot be "
        "explained by discarding the identity path or by a rank-256 ceiling. The "
        "remaining alternatives are context-conditioned/routed suffix maps, a "
        "different nonlinear shared parameterization, or insufficient discovery "
        "coverage—not one small static linear pullback.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
