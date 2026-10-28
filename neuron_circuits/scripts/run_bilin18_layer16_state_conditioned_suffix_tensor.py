#!/usr/bin/env python3
"""Fit a compact CP tensor from output covector and local state to suffix adjoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_output_conditioned_suffix_atlas import (
    COMPONENTS,
    behavior_fidelity,
    factor_recomposition_error,
    fidelity,
    pca_basis,
    shared_scaled_identity,
)


CONFIG = Path("configs/bilin18_layer16_state_conditioned_suffix_tensor.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_state_conditioned_suffix_tensor.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_STATE_CONDITIONED_SUFFIX_TENSOR.md")
MODEL_OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_state_conditioned_suffix_tensor.pt")


class ConditionalCP(torch.nn.Module):
    def __init__(self, query_dimension, state_dimension, rank, output_dimension, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.query = torch.nn.Parameter(
            torch.randn(query_dimension, rank, generator=generator) * .03)
        self.state = torch.nn.Parameter(
            torch.randn(state_dimension + 1, rank, generator=generator) * .03)
        self.output = torch.nn.Parameter(
            torch.randn(rank, output_dimension, generator=generator) * .001)

    def forward(self, query, state):
        state_with_constant = torch.cat(
            (state, torch.ones(len(state), 1, device=state.device)), dim=1)
        coefficients = (query @ self.query) * (state_with_constant @ self.state)
        return coefficients @ self.output


def make_balanced_rows(data, components, indices, component_scales):
    queries = []
    targets = []
    states = []
    for component in components:
        scale = component_scales[component]
        queries.append(data["components"][component]["output_covector"][indices].float() * scale)
        targets.append(data["components"][component]["write_adjoint"][indices].float() * scale)
        states.append(data["state"][indices].float())
    return torch.cat(queries), torch.cat(targets), torch.cat(states)


def chart_features(query, state, charts):
    query_features = (query.float() @ charts["query_basis"]) / charts["query_scale"]
    state_features = (
        ((state.float() - charts["state_mean"]) @ charts["state_basis"])
        / charts["state_scale"])
    return query_features, state_features


@torch.inference_mode()
def predict(model, query, state, charts):
    device = next(model.parameters()).device
    query_features, state_features = chart_features(query, state, charts)
    residual = model(
        query_features.to(device), state_features.to(device)).cpu() / charts["target_scale"]
    return charts["identity_scale"] * query.float() + residual


@torch.inference_mode()
def validation_fidelity(model, rows, charts):
    query, target, state = rows
    return fidelity(target, predict(model, query, state, charts))


def train_seed(train_rows, validation_rows, charts, config, seed):
    model = ConditionalCP(
        config["query_dimension"], config["state_dimension"], config["cp_rank"],
        train_rows[1].shape[1], seed).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"] + seed)
    train_query_features, train_state_features = chart_features(
        train_rows[0], train_rows[2], charts)
    target_residual = (
        train_rows[1] - charts["identity_scale"] * train_rows[0]) * charts["target_scale"]
    initial_validation = validation_fidelity(model, validation_rows, charts)
    best_validation = initial_validation
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    curve = [{"step": 0, "validation_write_fidelity": initial_validation}]
    for step in range(1, config["steps"] + 1):
        indices = torch.randint(
            0, len(train_rows[0]), (config["batch_size"],), generator=generator)
        query = train_query_features[indices].cuda()
        state = train_state_features[indices].cuda()
        target = target_residual[indices].cuda()
        prediction = model(query, state)
        loss = (target - prediction).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % config["validation_interval"] == 0:
            value = validation_fidelity(model, validation_rows, charts)
            curve.append({
                "step": step,
                "validation_write_fidelity": value,
                "minibatch_loss": float(loss.detach()),
            })
            if value > best_validation:
                best_validation = value
                best_step = step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    return model, {
        "initial_validation_write_fidelity": initial_validation,
        "best_validation_write_fidelity": best_validation,
        "best_step": best_step,
        "curve": curve,
    }


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    static_result = json.loads(Path(config["static_result_path"]).read_text())
    discovery = factors["discovery"]
    evaluations = factors["evaluation"]
    weights = factors["layer16_weights"]
    train_end = config["train_sites"]
    validation_end = train_end + config["validation_sites"]
    assert validation_end == len(discovery["state"])
    train_indices = slice(0, train_end)
    validation_indices = slice(train_end, validation_end)
    components = config["fit_components"]

    component_scales = {}
    for component in components:
        target = discovery["components"][component]["write_adjoint"][train_indices].double()
        component_scales[component] = float(
            (len(target) / target.square().sum().clamp_min(1e-30)).sqrt())
    train_rows = make_balanced_rows(
        discovery, components, train_indices, component_scales)
    validation_rows = make_balanced_rows(
        discovery, components, validation_indices, component_scales)
    identity_scale = shared_scaled_identity(train_rows[0].double(), train_rows[1].double())

    query_basis = pca_basis(train_rows[0], config["query_dimension"])
    state_train = discovery["state"][train_indices].float()
    state_mean = state_train.mean(0)
    state_basis = pca_basis(state_train - state_mean, config["state_dimension"])
    raw_query_features = train_rows[0] @ query_basis
    raw_state_features = (state_train - state_mean) @ state_basis
    query_scale = raw_query_features.square().mean(0).sqrt().clamp_min(1e-6)
    state_scale = raw_state_features.square().mean(0).sqrt().clamp_min(1e-6)
    residual = train_rows[1] - identity_scale * train_rows[0]
    target_scale = float((residual.numel() / residual.double().square().sum()).sqrt())
    charts = {
        "query_basis": query_basis,
        "query_scale": query_scale,
        "state_mean": state_mean,
        "state_basis": state_basis,
        "state_scale": state_scale,
        "target_scale": target_scale,
        "identity_scale": identity_scale,
    }

    seed_models = {}
    training = {}
    selected_seed = None
    selected_model = None
    selected_validation = -float("inf")
    for seed in config["training_seeds"]:
        print("train seed", seed, flush=True)
        model, record = train_seed(train_rows, validation_rows, charts, config, seed)
        seed_models[str(seed)] = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        training[str(seed)] = record
        if record["best_validation_write_fidelity"] > selected_validation:
            selected_validation = record["best_validation_write_fidelity"]
            selected_seed = seed
            selected_model = model

    generator = torch.Generator().manual_seed(config["seed"])
    results = {}
    for corpus_index, (corpus, data) in enumerate(evaluations.items()):
        permutation = torch.randperm(
            len(data["state"]), generator=torch.Generator().manual_seed(config["seed"] + corpus_index))
        rows = {}
        for component in COMPONENTS:
            query = data["components"][component]["output_covector"]
            native_state = data["state"]
            prediction = predict(selected_model, query, native_state, charts)
            shuffled_prediction = predict(
                selected_model, query, native_state[permutation], charts)
            rows[component] = {
                "write_fidelity": fidelity(
                    data["components"][component]["write_adjoint"], prediction),
                "shuffled_state_write_fidelity": fidelity(
                    data["components"][component]["write_adjoint"], shuffled_prediction),
                "behavior_fidelity": behavior_fidelity(
                    data, component, prediction, weights),
                "shuffled_state_behavior_fidelity": behavior_fidelity(
                    data, component, shuffled_prediction, weights),
                "static_behavior_fidelity": static_result["results"][corpus]["components"][component]["shared_behavior_fidelity"],
            }
        results[corpus] = {"sites": len(data["state"]), "components": rows}

    maximum_recomposition = max(
        factor_recomposition_error(discovery),
        *(factor_recomposition_error(data) for data in evaluations.values()),
    )
    validation_values = sorted(
        row["best_validation_write_fidelity"] for row in training.values())
    static_validation = static_result["shared_selection"]["mean_validation_write_fidelity"]
    tested_components = ("prediction", "target", "margin")
    hypotheses = {
        "T0_saved_factor_recomposition_is_exact": maximum_recomposition <= 1e-6,
        "T1_state_conditioning_improves_validation_write": selected_validation >= static_validation + .10,
        "T2_state_conditioned_tensor_predicts_behavior": all(
            row["components"][component]["behavior_fidelity"] >= .80
            for row in results.values() for component in tested_components),
        "T3_state_conditioned_tensor_explains_model_margin": all(
            row["components"]["margin"]["behavior_fidelity"] >= .80
            for row in results.values()),
        "T4_state_is_causally_specific_for_routing": all(
            row["components"][component]["behavior_fidelity"]
            >= row["components"][component]["shuffled_state_behavior_fidelity"] + .10
            for row in results.values() for component in tested_components),
        "T5_state_conditioned_tensor_beats_static_behavior": all(
            row["components"][component]["behavior_fidelity"]
            >= row["components"][component]["static_behavior_fidelity"] + .10
            for row in results.values() for component in tested_components),
        "T6_state_conditioned_tensor_transfers_ood": all(
            results[name]["components"][component]["behavior_fidelity"]
            >= results["fineweb"]["components"][component]["behavior_fidelity"] - .10
            for name in ("wikitext", "tinystories") for component in tested_components),
        "T7_training_seeds_are_stable": validation_values[-1] <= validation_values[1] + .05,
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["T0_saved_factor_recomposition_is_exact"]["maximum_relative_error"] = maximum_recomposition
    parameter_count = sum(parameter.numel() for parameter in selected_model.parameters())
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "parameter_count": parameter_count,
        "identity_scale": identity_scale,
        "target_scale": target_scale,
        "static_validation_write_fidelity": static_validation,
        "selected_seed": selected_seed,
        "selected_validation_write_fidelity": selected_validation,
        "training": training,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save({
        "charts": charts,
        "selected_seed": selected_seed,
        "selected_model": seed_models[str(selected_seed)],
        "all_models": seed_models,
    }, MODEL_OUTPUT)

    lines = [
        "# State-conditioned CP suffix pullback",
        "",
        "A rank-64 CP tensor predicts the write-adjoint correction from a 256D "
        "output-covector chart and a 64D centered local-state chart. It is shared "
        "across prediction, target, and margin endpoints and sits on top of the "
        "fitted scaled identity. Three seeds use validation-only checkpoint selection.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Parameters: `{parameter_count:,}`. Static validation write fidelity: "
        f"`{static_validation:.3f}`; selected CP validation fidelity: "
        f"`{selected_validation:.3f}` (seed `{selected_seed}`).",
        "",
        "| Seed | Initial validation | Best validation | Step |",
        "|---:|---:|---:|---:|",
    ]
    for seed in config["training_seeds"]:
        row = training[str(seed)]
        lines.append(
            f"| {seed} | {row['initial_validation_write_fidelity']:.3f} | "
            f"{row['best_validation_write_fidelity']:.3f} | {row['best_step']} |")
    lines += [
        "",
        "| Corpus | Component | Write | Shuffled-state write | Behavior | Shuffled-state behavior | Static behavior |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for component in COMPONENTS:
            item = row["components"][component]
            lines.append(
                f"| {corpus} | {component} | {item['write_fidelity']:.3f} | "
                f"{item['shuffled_state_write_fidelity']:.3f} | "
                f"{item['behavior_fidelity']:.3f} | "
                f"{item['shuffled_state_behavior_fidelity']:.3f} | "
                f"{item['static_behavior_fidelity']:.3f} |")
    lines += [
        "",
        "State-shuffle specificity distinguishes a genuinely routed tensor from an "
        "overparameterized q-only map. Failure remains scoped: Layer-17 attention "
        "also reads sequence-wide keys, values, and the persistent embedding bus, "
        "which are not explicitly supplied by this local-state router.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
