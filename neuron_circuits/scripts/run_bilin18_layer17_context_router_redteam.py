#!/usr/bin/env python3
"""Red-team lexical routing of the compact Layer-17 tangent operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token
from run_bilin18_layer17_context_operator_atlas import (
    collect,
    fit_lookup,
    lookup_predict,
    operator_fidelity,
    operator_rows,
)
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor, fidelity
from run_bilin18_layer17_tangent_output_semantics import vocabulary_table
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_context_router_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_router_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_ROUTER_REDTEAM.md")


def gram_fidelity(target, prediction, gram, mask=None):
    if mask is not None:
        target = target[mask]
        prediction = prediction[mask]
    difference = (target - prediction).double()
    target = target.double()
    gram = gram.double()
    error = torch.einsum("bi,ij,bj->", difference, gram, difference)
    energy = torch.einsum("bi,ij,bj->", target, gram, target).clamp_min(1e-30)
    return 1.0 - float(error / energy)


def position_indices(position, positions):
    mapping = {value: index for index, value in enumerate(positions)}
    return np.asarray([mapping[int(value)] for value in position], dtype=np.int64)


def coarse_ids(tokenizer, ids):
    return np.asarray(
        [classify_token(tokenizer.decode([int(index)])) for index in ids],
        dtype=np.int64,
    )


def ridge_fit(x, y, relative_penalty):
    x = x.float().cuda()
    y = y.float().cuda()
    x_mean = x.mean(0)
    y_mean = y.mean(0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    penalty = relative_penalty * float(gram.trace() / gram.shape[0])
    system = gram + penalty * torch.eye(gram.shape[0], device=gram.device)
    weight = torch.linalg.solve(system, xc.T @ yc)
    return x_mean.cpu(), y_mean.cpu(), weight.cpu(), penalty


def ridge_predict(x, fit):
    x_mean, y_mean, weight, _ = fit
    return (x.float().cpu() - x_mean) @ weight + y_mean


def matched_groups(tokenizer, counts, cap):
    groups = {}
    for token in torch.nonzero(counts > 0).flatten().tolist():
        coarse = classify_token(tokenizer.decode([int(token)]))
        frequency = min(int(np.floor(np.log2(float(counts[token])))), cap)
        groups.setdefault((coarse, frequency), []).append(token)
    return [np.asarray(group, dtype=np.int64) for group in groups.values() if len(group) > 1]


def matched_nulls(table, counts, evaluation, tokenizer, gram, config, seed):
    groups = matched_groups(tokenizer, counts, config["frequency_bin_cap"])
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(config["matched_permutations"]):
        permutation = torch.arange(len(table))
        for group in groups:
            permutation[torch.as_tensor(group)] = torch.as_tensor(generator.permutation(group))
        prediction = table[permutation[torch.as_tensor(evaluation["current"], dtype=torch.long)]]
        values.append(gram_fidelity(evaluation["coefficients"], prediction, gram))
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, .95)),
        "maximum": float(np.max(values)),
        "replicates": len(values),
    }


def router_bundle(data, tables, tokenizer, embeddings, ridge, gram, config):
    positions = config["positions"]
    position_id = position_indices(data["position"], positions)
    coarse = coarse_ids(tokenizer, data["current"])
    token_position = data["current"] * len(positions) + position_id
    coarse_position = coarse * len(positions) + position_id
    predictions = {
        "mean": tables["mean"][None].expand(len(data["coefficients"]), -1),
        "position": tables["position"][position_id],
        "coarse": tables["coarse"][coarse],
        "coarse_position": tables["coarse_position"][coarse_position],
        "token": tables["token"][torch.as_tensor(data["current"], dtype=torch.long)],
        "token_position": tables["token_position"][torch.as_tensor(token_position, dtype=torch.long)],
        "embedding_ridge": ridge_predict(embeddings[torch.as_tensor(data["current"], device="cuda")], ridge),
    }
    metrics = {
        name: {
            "coordinate_fidelity": fidelity(data["coefficients"], prediction),
            "operator_fidelity": gram_fidelity(data["coefficients"], prediction, gram),
        }
        for name, prediction in predictions.items()
    }
    counts = tables["token_counts"][torch.as_tensor(data["current"], dtype=torch.long)]
    curves = {}
    for threshold in config["minimum_train_counts"]:
        mask = counts >= threshold
        curves[str(threshold)] = {
            "examples": int(mask.sum()),
            "fraction": float(mask.double().mean()),
            **{
                name: gram_fidelity(data["coefficients"], predictions[name], gram, mask)
                for name in ("mean", "coarse", "token", "embedding_ridge")
            },
        }
    metrics["count_curve"] = curves
    metrics["token_coverage"] = float((counts > 0).double().mean())
    return metrics, predictions


def cross_position(data, early_table, early_counts, late_table, late_counts, gram, config):
    position = data["position"]
    early_mask = np.isin(position, config["early_positions"])
    late_mask = np.isin(position, config["late_positions"])
    early_prediction, _ = lookup_predict(data["current"], early_table, early_counts)
    late_prediction, _ = lookup_predict(data["current"], late_table, late_counts)
    rows = {
        "early_to_late": {
            "operator_fidelity": gram_fidelity(data["coefficients"], early_prediction, gram, late_mask),
            "coverage": float((early_counts[torch.as_tensor(data["current"][late_mask])] > 0).double().mean()),
        },
        "late_to_early": {
            "operator_fidelity": gram_fidelity(data["coefficients"], late_prediction, gram, early_mask),
            "coverage": float((late_counts[torch.as_tensor(data["current"][early_mask])] > 0).double().mean()),
        },
    }
    directions = list(rows.values())
    rows["mean_fidelity"] = float(np.mean([row["operator_fidelity"] for row in directions]))
    rows["mean_coverage"] = float(np.mean([row["coverage"] for row in directions]))
    return rows


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    _, vocabulary, base_names = vocabulary_table(tokenizer)
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
    flat_tensor = tensor.reshape(-1, config["rank"]).double()
    gram = flat_tensor.T @ flat_tensor

    print("collect discovery", flush=True)
    discovery = collect(model, tokenizer, config["discovery"], config, context_axes, vocabulary, base_names)
    selection = discovery["row"] < config["discovery"]["rows"] // 2
    train_coefficients = discovery["coefficients"][selection]
    train_tokens = discovery["current"][selection]
    train_positions = discovery["position"][selection]
    position_id = position_indices(train_positions, config["positions"])
    train_coarse = coarse_ids(tokenizer, train_tokens)

    token_table, token_counts, mean = fit_lookup(train_tokens, train_coefficients, tokenizer.vocab_size)
    position_table, _, _ = fit_lookup(position_id, train_coefficients, len(config["positions"]))
    coarse_table, _, _ = fit_lookup(train_coarse, train_coefficients, len(CLASSES))
    coarse_position_table, _, _ = fit_lookup(
        train_coarse * len(config["positions"]) + position_id,
        train_coefficients, len(CLASSES) * len(config["positions"]))
    token_position_table, _, _ = fit_lookup(
        train_tokens * len(config["positions"]) + position_id,
        train_coefficients, tokenizer.vocab_size * len(config["positions"]))
    tables = {
        "mean": mean,
        "token": token_table,
        "token_counts": token_counts,
        "position": position_table,
        "coarse": coarse_table,
        "coarse_position": coarse_position_table,
        "token_position": token_position_table,
    }

    embeddings = model.embedding.weight.float().detach()
    ridge = ridge_fit(
        embeddings[torch.as_tensor(train_tokens, device="cuda")],
        train_coefficients, config["ridge_relative_penalty"])

    early_train = selection & np.isin(discovery["position"], config["early_positions"])
    late_train = selection & np.isin(discovery["position"], config["late_positions"])
    early_table, early_counts, _ = fit_lookup(
        discovery["current"][early_train], discovery["coefficients"][early_train], tokenizer.vocab_size)
    late_table, late_counts, _ = fit_lookup(
        discovery["current"][late_train], discovery["coefficients"][late_train], tokenizer.vocab_size)

    results = {}
    explicit_error = None
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect(model, tokenizer, spec, config, context_axes, vocabulary, base_names)
        metrics, predictions = router_bundle(data, tables, tokenizer, embeddings, ridge, gram, config)
        metrics["matched_identity_null"] = matched_nulls(
            token_table, token_counts, data, tokenizer, gram, config,
            config["seed"] + list(config["evaluation"]).index(corpus))
        metrics["incremental_token_over_coarse"] = float(
            (metrics["token"]["operator_fidelity"] - metrics["coarse"]["operator_fidelity"])
            / max(1e-30, 1 - metrics["coarse"]["operator_fidelity"]))
        metrics["cross_position"] = cross_position(
            data, early_table, early_counts, late_table, late_counts, gram, config)
        if explicit_error is None:
            count = min(2048, len(data["coefficients"]))
            target = operator_rows(data["coefficients"][:count], tensor)
            prediction = operator_rows(predictions["token"][:count], tensor)
            explicit = operator_fidelity(target, prediction)
            implicit = gram_fidelity(
                data["coefficients"][:count], predictions["token"][:count], gram)
            explicit_error = abs(explicit - implicit)
        results[corpus] = metrics

    hypotheses = {
        "R0_tensor_metric_matches_explicit_operator": explicit_error <= 1e-6,
        "R1_exact_token_beats_matched_identity_null": all(
            row["token"]["operator_fidelity"] >= row["matched_identity_null"]["p95"] + .03
            for row in results.values()),
        "R2_exact_token_adds_beyond_coarse_class": all(
            row["incremental_token_over_coarse"] >= .05 for row in results.values()),
        "R3_repeated_tokens_strengthen_identity_routing": all(
            row["count_curve"]["4"]["token"] >= row["count_curve"]["4"]["coarse"] + .05
            for row in results.values()),
        "R4_token_routing_transfers_across_position_halves": all(
            row["cross_position"]["mean_fidelity"] >= .75
            and row["cross_position"]["mean_coverage"] >= .40 for row in results.values()),
        "R5_embedding_geometry_routes_unseen_contexts": all(
            row["embedding_ridge"]["operator_fidelity"] >= .75 for row in results.values()),
        "R6_position_alone_is_not_the_router": all(
            row["position"]["operator_fidelity"] <= row["token"]["operator_fidelity"] - .05
            for row in results.values()),
        "R7_coarse_position_does_not_explain_exact_identity": all(
            row["coarse_position"]["operator_fidelity"] <= row["token"]["operator_fidelity"] - .03
            for row in results.values()),
        "R8_token_position_lookup_is_not_required": all(
            row["token"]["operator_fidelity"] >= row["token_position"]["operator_fidelity"] - .05
            for row in results.values()),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "tensor_metric_error": explicit_error,
        "ridge_penalty": ridge[3],
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 context-router red team", "",
        "The compact context-conditioned operator is retained. This test asks whether its apparent current-token routing survives stronger lexical, frequency, position, and embedding controls.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Tensor-Gram versus explicit-operator fidelity error: `{explicit_error:.2e}`.",
              "", "## Router comparison", "",
              "| Corpus | mean / position | coarse / coarse+position | exact token / token+position | embedding ridge | matched null p95 | token residual gain | cross-position fidelity/coverage |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        op = lambda name: row[name]["operator_fidelity"]
        lines.append(
            f"| {corpus} | {op('mean'):.3f}/{op('position'):.3f} | "
            f"{op('coarse'):.3f}/{op('coarse_position'):.3f} | "
            f"{op('token'):.3f}/{op('token_position'):.3f} | {op('embedding_ridge'):.3f} | "
            f"{row['matched_identity_null']['p95']:.3f} | {row['incremental_token_over_coarse']:.3f} | "
            f"{row['cross_position']['mean_fidelity']:.3f}/{row['cross_position']['mean_coverage']:.3f} |")
    lines += ["", "## Training-count conditional fidelity", "",
              "Each cell is `examples: coarse / exact-token / embedding-ridge` in the tensor-induced operator metric.", ""]
    for corpus, row in results.items():
        lines.append(f"### {corpus}")
        lines.append("")
        lines.append("| Minimum discovery count | Examples | Coarse | Exact token | Embedding ridge |")
        lines.append("|---:|---:|---:|---:|---:|")
        for threshold, curve in row["count_curve"].items():
            lines.append(
                f"| {threshold} | {curve['examples']} | {curve['coarse']:.3f} | "
                f"{curve['token']:.3f} | {curve['embedding_ridge']:.3f} |")
        lines.append("")
    lines += [
        "## Interpretation boundary", "",
        "Exact-token success would identify a reusable lexical prior over local operators, not a complete context-independent instruction: the remaining error still depends on the full hidden state. Embedding-ridge success would show that this prior extends through learned lexical geometry rather than only repeated token IDs. The tensor core and context subspace remain basis-invariant even if every lexical router fails.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "tensor_metric_error": explicit_error,
        "summary": {corpus: {
            "mean": row["mean"]["operator_fidelity"],
            "position": row["position"]["operator_fidelity"],
            "coarse": row["coarse"]["operator_fidelity"],
            "coarse_position": row["coarse_position"]["operator_fidelity"],
            "token": row["token"]["operator_fidelity"],
            "token_position": row["token_position"]["operator_fidelity"],
            "embedding": row["embedding_ridge"]["operator_fidelity"],
            "null_p95": row["matched_identity_null"]["p95"],
            "cross_position": row["cross_position"],
        } for corpus, row in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
