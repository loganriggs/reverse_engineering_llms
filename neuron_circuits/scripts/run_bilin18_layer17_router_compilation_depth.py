#!/usr/bin/env python3
"""Trace when the final compact operator router becomes causally committed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER, classify_token
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_layer17_tangent_output_semantics import vocabulary_table
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_router_compilation_depth.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_router_compilation_depth.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_ROUTER_COMPILATION_DEPTH.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_router_compilation_depth.png")


def prepare_ridge(x, relative_penalty):
    x = x.float().cuda()
    scale = float(x.double().square().mean().sqrt().clamp_min(1e-30))
    x = x / scale
    mean = x.mean(0)
    centered = x - mean
    gram = centered.T @ centered
    gram = 0.5 * (gram + gram.T)
    penalty = relative_penalty * float(gram.trace() / gram.shape[0])
    cholesky = torch.linalg.cholesky(
        gram + penalty * torch.eye(gram.shape[0], device="cuda"))
    return mean, centered, cholesky, penalty, scale


def fit_response(prepared, y):
    x_mean, centered, cholesky, _, scale = prepared
    y = y.float().cuda()
    y_mean = y.mean(0)
    weight = torch.cholesky_solve(centered.T @ (y - y_mean), cholesky)
    return {
        "x_mean": x_mean.cpu(), "y_mean": y_mean.cpu(), "weight": weight.cpu(),
        "x_scale": scale,
    }


def predict(x, fit):
    return (x.float() / fit["x_scale"] - fit["x_mean"]) @ fit["weight"] + fit["y_mean"]


def gram_fidelity(target, prediction, gram):
    target = target.double(); prediction = prediction.double(); difference = target - prediction
    error = torch.einsum("bi,ij,bj->", difference, gram, difference)
    energy = torch.einsum("bi,ij,bj->", target, gram, target).clamp_min(1e-30)
    return 1.0 - float(error / energy)


def coarse_ids(tokenizer, ids):
    return np.asarray(
        [classify_token(tokenizer.decode([int(index)])) for index in ids], dtype=np.int64)


def position_indices(position, positions):
    mapping = {value: index for index, value in enumerate(positions)}
    return np.asarray([mapping[int(value)] for value in position], dtype=np.int64)


def matched_response_permutation(coarse, position, generator):
    permutation = np.arange(len(coarse))
    for coarse_value in np.unique(coarse):
        for position_value in np.unique(position):
            group = np.flatnonzero((coarse == coarse_value) & (position == position_value))
            if len(group) > 1:
                permutation[group] = generator.permutation(group)
    return permutation


@torch.inference_mode()
def collect_prediction(model, spec, config, context_axes):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = np.asarray(config["prediction_positions"])
    position_tensor = torch.as_tensor(positions, device="cuda")
    states = {layer: [] for layer in config["prediction_boundaries"]}
    embeddings, coefficients, tokens, position_rows = [], [], [], []
    dimension = model.config.n_embd
    for start in range(0, len(rows), config["prediction_batch_size"]):
        raw = np.asarray(rows[start:start + config["prediction_batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        embeddings.append(x0[:, position_tensor].flatten(0, 1).bfloat16().cpu())
        x = x0
        value_bus = None
        for layer, block in enumerate(model.blocks[:17]):
            x, value_bus = block(x, value_bus, x0)
            if layer in states:
                states[layer].append(x[:, position_tensor].flatten(0, 1).bfloat16().cpu())
        block = model.blocks[17]
        mixed = block.lambdas[0].float() * x.float() + block.lambdas[1].float() * x0.float()
        attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        final_state = F.rms_norm(mixed + attention.float(), (dimension,))
        coefficients.append((final_state[:, position_tensor] @ context_axes).flatten(0, 1).cpu())
        batch = len(raw)
        tokens.extend(raw[:, positions].reshape(-1).tolist())
        position_rows.extend(np.tile(positions, batch).tolist())
    return {
        "states": {layer: torch.cat(values) for layer, values in states.items()},
        "embedding": torch.cat(embeddings),
        "coefficients": torch.cat(coefficients),
        "tokens": np.asarray(tokens),
        "position": np.asarray(position_rows),
    }


def fit_predictors(discovery, context_axes, config, tokenizer):
    target = discovery["coefficients"]
    embedding_prepared = prepare_ridge(discovery["embedding"], config["ridge_relative_penalty"])
    embedding_fit = fit_response(embedding_prepared, target)
    full, context = {}, {}
    layer16_prepared = None
    for layer in config["prediction_boundaries"]:
        print("fit predictor", layer, flush=True)
        x = discovery["states"][layer]
        prepared = prepare_ridge(x, config["ridge_relative_penalty"])
        full[layer] = fit_response(prepared, target)
        if layer == 16:
            layer16_prepared = prepared
        else:
            del prepared
        projected = x.float().cuda() @ context_axes
        restricted = prepare_ridge(projected.cpu(), config["ridge_relative_penalty"])
        context[layer] = fit_response(restricted, target)
        del restricted, projected
        torch.cuda.empty_cache()

    prepared = layer16_prepared
    coarse = coarse_ids(tokenizer, discovery["tokens"])
    position = position_indices(discovery["position"], config["prediction_positions"])
    generator = np.random.default_rng(config["seed"])
    nulls = []
    for _ in range(config["response_permutations"]):
        permutation = matched_response_permutation(coarse, position, generator)
        nulls.append(fit_response(prepared, target[permutation]))
    del prepared
    return {"embedding": embedding_fit, "full": full, "context": context, "nulls": nulls}


def evaluate_predictors(data, fits, context_axes, gram, config):
    target = data["coefficients"]
    embedding_prediction = predict(data["embedding"], fits["embedding"])
    output = {"embedding": gram_fidelity(target, embedding_prediction, gram), "layers": {}}
    for layer in config["prediction_boundaries"]:
        x = data["states"][layer]
        full = predict(x, fits["full"][layer])
        projected = x.float() @ context_axes.cpu()
        context = predict(projected, fits["context"][layer])
        output["layers"][str(layer)] = {
            "full_state": gram_fidelity(target, full, gram),
            "context_span": gram_fidelity(target, context, gram),
        }
    null_values = [
        gram_fidelity(target, predict(data["states"][16], fit), gram)
        for fit in fits["nulls"]
    ]
    output["layer16_response_null"] = {
        "mean": float(np.mean(null_values)),
        "p95": float(np.quantile(null_values, .95)),
        "maximum": float(np.max(null_values)),
        "replicates": len(null_values),
    }
    return output


def discovery_means(discovery, config):
    means = {}
    for layer in config["causal_boundaries"]:
        layer_means = []
        for position in config["causal_positions"]:
            mask = discovery["position"] == position
            layer_means.append(discovery["states"][layer][mask].float().mean(0))
        means[layer] = torch.stack(layer_means).cuda()
    return means


@torch.inference_mode()
def trace_boundaries(model, ids, boundaries):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    states = {}
    for layer, block in enumerate(model.blocks[:17]):
        x, value_bus = block(x, value_bus, x0)
        if layer in boundaries:
            states[layer] = x.float()
    return x0.float(), value_bus.float(), states


@torch.inference_mode()
def suffix_state(model, x, value_bus, x0, boundary):
    dimension = model.config.n_embd
    for layer in range(boundary + 1, 17):
        x, value_bus = model.blocks[layer](x, value_bus, x0)
    block = model.blocks[17]
    mixed = block.lambdas[0].float() * x.float() + block.lambdas[1].float() * x0.float()
    attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    return F.rms_norm(mixed + attention.float(), (dimension,))


@torch.inference_mode()
def causal_trace(model, spec, means, context_axes, gram, config):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    boundaries = config["causal_boundaries"]
    positions = config["causal_positions"]
    target_rows = {layer: [] for layer in boundaries}
    mean_rows = {layer: [] for layer in boundaries}
    cycle_rows = {layer: [] for layer in boundaries}
    exact_error = 0.0; exact_energy = 0.0
    for start in range(0, len(rows), config["causal_batch_size"]):
        raw = np.asarray(rows[start:start + config["causal_batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, states = trace_boundaries(model, ids, set(boundaries))
        _, reference = prefix_to_layer17(model, ids)
        native = suffix_state(model, states[16], value_bus, x0, 16)
        reference_selected = torch.cat([reference[:, position] @ context_axes for position in positions])
        native_selected = torch.cat([native[:, position] @ context_axes for position in positions])
        exact_error += float((reference_selected.double() - native_selected.double()).square().sum())
        exact_energy += float(reference_selected.double().square().sum())
        batch = len(ids)
        target = reference_selected.cpu()
        for boundary in boundaries:
            chunks = []
            for method in ("mean", "cycle"):
                for index, position in enumerate(positions):
                    edited = states[boundary].clone()
                    if method == "mean":
                        edited[:, position] = means[boundary][index]
                    else:
                        edited[:, position] = torch.roll(states[boundary][:, position], 1, 0)
                    chunks.append(edited)
            variants = torch.cat(chunks)
            repeats = len(chunks)
            repeat_x0 = x0.repeat(repeats, 1, 1)
            repeat_bus = value_bus.repeat(*([repeats] + [1] * (value_bus.ndim - 1)))
            final = suffix_state(model, variants, repeat_bus, repeat_x0, boundary)
            position_vector = torch.as_tensor(
                [position for _ in ("mean", "cycle") for position in positions for _ in range(batch)],
                device="cuda")
            selected = final[torch.arange(len(final), device="cuda"), position_vector] @ context_axes
            split = len(positions) * batch
            target_rows[boundary].append(target)
            mean_rows[boundary].append(selected[:split].cpu())
            cycle_rows[boundary].append(selected[split:].cpu())
    results = {}
    for boundary in boundaries:
        target = torch.cat(target_rows[boundary])
        results[str(boundary)] = {
            "mean_replacement": gram_fidelity(target, torch.cat(mean_rows[boundary]), gram),
            "cyclic_substitution": gram_fidelity(target, torch.cat(cycle_rows[boundary]), gram),
            "examples": len(target),
        }
    return results, float(np.sqrt(exact_error / max(exact_energy, 1e-30)))


def plot_results(prediction, causal, config):
    corpora = list(prediction)
    colors = {corpus: color for corpus, color in zip(corpora, ("#2563eb", "#dc2626", "#16a34a"))}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    depths = config["prediction_boundaries"]
    for corpus in corpora:
        axes[0].plot(depths, [prediction[corpus]["layers"][str(d)]["full_state"] for d in depths],
                     marker="o", color=colors[corpus], label=corpus)
        axes[1].plot(depths, [prediction[corpus]["layers"][str(d)]["context_span"] for d in depths],
                     marker="o", color=colors[corpus], label=corpus)
    causal_depths = config["causal_boundaries"]
    for corpus in corpora:
        axes[2].plot(causal_depths, [causal[corpus][str(d)]["mean_replacement"] for d in causal_depths],
                     marker="o", color=colors[corpus], label=f"{corpus} mean")
        axes[2].plot(causal_depths, [causal[corpus][str(d)]["cyclic_substitution"] for d in causal_depths],
                     marker="x", linestyle="--", color=colors[corpus], alpha=.75)
    axes[0].set_title("Full-state predictive chart")
    axes[1].set_title("Context-span predictive chart")
    axes[2].set_title("Finite causal replacement")
    for axis in axes:
        axis.set_xlabel("block boundary")
        axis.set_ylabel("operator fidelity")
        axis.axhline(.9, color="#6b7280", linestyle=":", linewidth=1)
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)
    flat = tensor.reshape(-1, config["rank"]).double()
    gram = flat.T @ flat

    print("collect discovery", flush=True)
    discovery = collect_prediction(model, config["discovery"], config, context_axes)
    fits = fit_predictors(discovery, context_axes, config, tokenizer)
    prediction = {}
    for corpus, spec in config["prediction_evaluation"].items():
        print("prediction", corpus, flush=True)
        data = collect_prediction(model, spec, config, context_axes)
        prediction[corpus] = evaluate_predictors(data, fits, context_axes, gram, config)
        del data

    means = discovery_means(discovery, config)
    causal = {}; exact_errors = {}
    for corpus, spec in config["causal_evaluation"].items():
        print("causal", corpus, flush=True)
        causal[corpus], exact_errors[corpus] = causal_trace(
            model, spec, means, context_axes, gram, config)

    first_below = {}
    for corpus, rows in causal.items():
        first_below[corpus] = next(
            (depth for depth in config["causal_boundaries"]
             if rows[str(depth)]["mean_replacement"] < .90), 18)
    hypotheses = {
        "D0_native_suffix_recomposition_is_exact": max(exact_errors.values()) <= 1e-6,
        "D1_final_carried_state_predicts_operator": all(
            row["layers"]["16"]["full_state"] >= .95 for row in prediction.values()),
        "D2_final_state_improves_on_embedding_chart": all(
            row["layers"]["16"]["full_state"] >= row["embedding"] + .05
            for row in prediction.values()),
        "D3_final_predictor_beats_matched_response_null": all(
            row["layers"]["16"]["full_state"] >= row["layer16_response_null"]["p95"] + .10
            for row in prediction.values()),
        "D4_early_state_is_causally_recoverable": all(
            row["0"]["mean_replacement"] >= .90 for row in causal.values()),
        "D5_final_state_is_causally_necessary": all(
            row["16"]["mean_replacement"] <= .80 for row in causal.values()),
        "D6_causal_dependence_grows_with_depth": all(
            row["0"]["mean_replacement"] - row["16"]["mean_replacement"] >= .15
            for row in causal.values()),
        "D7_cyclic_final_state_is_material": all(
            row["16"]["cyclic_substitution"] <= .80 for row in causal.values()),
        "D8_decodability_precedes_causal_commitment": all(any(
            prediction[corpus]["layers"][str(depth)]["full_state"] >= .80
            and causal[corpus][str(depth)]["mean_replacement"] >= .90
            for depth in config["causal_boundaries"]) for corpus in causal),
        "D9_compilation_boundary_transfers": max(first_below.values()) - min(first_below.values()) <= 4,
        "D10_context_span_chart_is_useful_at_the_end": all(
            row["layers"]["16"]["context_span"] >= .80 for row in prediction.values()),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "native_recomposition_errors": exact_errors,
        "first_mean_replacement_below_090": first_below,
        "prediction": prediction,
        "causal": causal,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    plot_results(prediction, causal, config)

    lines = [
        "# Causal compilation depth of the Layer-17 operator router", "",
        "Predictive charts and finite causal cuts are measured separately. The causal intervention replaces one current-position carried state, preserves the original embedding and persistent value bus, and exactly recomputes the remaining tensor-network suffix.",
        "", "![Router compilation depth](bilin18_layer17_router_compilation_depth.png)",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Predictive fidelity by block boundary", "",
              "| Layer | " + " | ".join(f"{corpus} full/context" for corpus in prediction) + " |",
              "|---:|" + "---:|" * len(prediction)]
    for depth in config["prediction_boundaries"]:
        values = []
        for corpus, row in prediction.items():
            layer = row["layers"][str(depth)]
            values.append(f"{layer['full_state']:.3f}/{layer['context_span']:.3f}")
        lines.append(f"| {depth} | " + " | ".join(values) + " |")
    lines += ["", "Native-embedding baseline: " + ", ".join(
        f"{corpus} `{row['embedding']:.3f}`" for corpus, row in prediction.items()) + ".",
        "", "## Finite causal fidelity", "",
        "| Layer | " + " | ".join(f"{corpus} mean/cycle" for corpus in causal) + " |",
        "|---:|" + "---:|" * len(causal)]
    for depth in config["causal_boundaries"]:
        values = [f"{row[str(depth)]['mean_replacement']:.3f}/{row[str(depth)]['cyclic_substitution']:.3f}"
                  for row in causal.values()]
        lines.append(f"| {depth} | " + " | ".join(values) + " |")
    lines += ["", "First tested mean-replacement boundary below 0.90: " + ", ".join(
        f"{corpus} Block {depth}" for corpus, depth in first_below.items()) + ".",
        "", "## Interpretation boundary", "",
        "A predictive chart can become accurate before its state is causally indispensable because later blocks retain the original embedding and can reconstruct the router. The finite cut measures recoverability under one intervention, not exclusive information storage. A gradual curve should not be renamed a discrete phase transition merely because a 0.90 reporting threshold is crossed."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "exact_errors": exact_errors,
        "first_below": first_below,
        "prediction": {corpus: {
            "embedding": row["embedding"],
            "layer0": row["layers"]["0"],
            "layer8": row["layers"]["8"],
            "layer16": row["layers"]["16"],
            "null_p95": row["layer16_response_null"]["p95"],
        } for corpus, row in prediction.items()},
        "causal": {corpus: {depth: row[depth] for depth in ("0", "8", "12", "14", "16")}
                   for corpus, row in causal.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
