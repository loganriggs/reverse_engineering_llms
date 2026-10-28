#!/usr/bin/env python3
"""Test whether Layer-17 embedding routing generalizes beyond seen token IDs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token
from run_bilin18_layer17_context_operator_atlas import collect, fit_lookup
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_layer17_tangent_output_semantics import vocabulary_table
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_embedding_router_redteam.json")
PRIOR = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_router_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_embedding_router_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_EMBEDDING_ROUTER_REDTEAM.md")


def gram_fidelity(target, prediction, gram, mask=None):
    target = target.double().cuda()
    prediction = prediction.double().cuda()
    gram = gram.double().cuda()
    if mask is not None:
        mask = torch.as_tensor(mask, dtype=torch.bool, device="cuda")
        target = target[mask]
        prediction = prediction[mask]
    difference = target - prediction
    error = torch.einsum("bi,ij,bj->", difference, gram, difference)
    energy = torch.einsum("bi,ij,bj->", target, gram, target).clamp_min(1e-30)
    return 1.0 - float(error / energy)


def prepare_ridge(x, relative_penalty):
    x = x.float().cuda()
    mean = x.mean(0)
    centered = x - mean
    gram = centered.T @ centered
    penalty = relative_penalty * float(gram.trace() / gram.shape[0])
    cholesky = torch.linalg.cholesky(
        gram + penalty * torch.eye(gram.shape[0], device="cuda"))
    return {"mean": mean, "centered": centered, "cholesky": cholesky, "penalty": penalty}


def fit_response(prepared, y):
    y = y.float().cuda()
    mean = y.mean(0)
    rhs = prepared["centered"].T @ (y - mean)
    weight = torch.cholesky_solve(rhs, prepared["cholesky"])
    return {"x_mean": prepared["mean"], "y_mean": mean, "weight": weight}


def predict(x, fit):
    return (x.float().cuda() - fit["x_mean"]) @ fit["weight"] + fit["y_mean"]


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


def subset_metrics(target, predictions, gram, seen):
    output = {}
    for subset, mask in (("all", None), ("seen", seen), ("unseen", ~seen)):
        output[subset] = {
            "examples": len(target) if mask is None else int(mask.sum()),
            **{name: gram_fidelity(target, value, gram, mask)
               for name, value in predictions.items()},
        }
    return output


def main():
    config = json.loads(CONFIG.read_text())
    prior = json.loads(PRIOR.read_text())
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
    flat = tensor.reshape(-1, config["rank"]).double()
    operator_gram = flat.T @ flat

    print("collect discovery", flush=True)
    discovery = collect(model, tokenizer, config["discovery"], config, context_axes, vocabulary, base_names)
    selection = discovery["row"] < config["discovery"]["rows"] // 2
    train_tokens = discovery["current"][selection]
    train_y = discovery["coefficients"][selection]
    train_coarse = coarse_ids(tokenizer, train_tokens)
    train_position = position_indices(discovery["position"][selection], config["positions"])
    token_table, token_counts, mean = fit_lookup(train_tokens, train_y, tokenizer.vocab_size)
    coarse_table, _, _ = fit_lookup(train_coarse, train_y, len(CLASSES))

    embeddings = model.embedding.weight.float().detach()
    train_embeddings = embeddings[torch.as_tensor(train_tokens, device="cuda")]
    full_prepared = prepare_ridge(train_embeddings, config["ridge_relative_penalty"])
    full_fit = fit_response(full_prepared, train_y)
    restricted_x = train_embeddings @ context_axes
    restricted_prepared = prepare_ridge(restricted_x, config["ridge_relative_penalty"])
    restricted_fit = fit_response(restricted_prepared, train_y)

    q_router = torch.linalg.qr(full_fit["weight"]).Q[:, :config["rank"]]
    canonical = torch.linalg.svdvals(q_router.T @ context_axes)
    alignment = {
        "canonical_cosines": [float(value) for value in canonical.cpu()],
        "mean_canonical_cosine": float(canonical.mean()),
        "projector_overlap": float(canonical.square().mean()),
    }
    u, singular, vh = torch.linalg.svd(full_fit["weight"], full_matrices=False)
    rank_fits = {}
    for rank in config["map_ranks"]:
        rank_fits[str(rank)] = {
            **full_fit,
            "weight": (u[:, :rank] * singular[:rank]) @ vh[:rank],
        }

    evaluations = {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect(model, tokenizer, spec, config, context_axes, vocabulary, base_names)
        ids = torch.as_tensor(data["current"], dtype=torch.long)
        x = embeddings[ids.cuda()]
        coarse = coarse_ids(tokenizer, data["current"])
        predictions = {
            "mean": mean[None].expand(len(ids), -1),
            "coarse": coarse_table[torch.as_tensor(coarse)],
            "token_table": token_table[ids],
            "embedding_full": predict(x, full_fit),
            "embedding_context_span": predict(x @ context_axes, restricted_fit),
        }
        for rank, fit in rank_fits.items():
            predictions[f"embedding_rank_{rank}"] = predict(x, fit)
        seen = (token_counts[ids] > 0).numpy()
        evaluations[corpus] = {
            "data": data,
            "x": x,
            "seen": seen,
            "metrics": subset_metrics(data["coefficients"], predictions, operator_gram, seen),
        }

    generator = np.random.default_rng(config["seed"])
    null_values = {corpus: [] for corpus in evaluations}
    for index in range(config["response_permutations"]):
        permutation = matched_response_permutation(train_coarse, train_position, generator)
        null_fit = fit_response(full_prepared, train_y[permutation])
        for corpus, row in evaluations.items():
            prediction = predict(row["x"], null_fit)
            null_values[corpus].append(gram_fidelity(
                row["data"]["coefficients"], prediction, operator_gram, ~row["seen"]))
        if (index + 1) % 8 == 0:
            print("null", index + 1, flush=True)

    results = {}
    for corpus, row in evaluations.items():
        values = np.asarray(null_values[corpus])
        metrics = row["metrics"]
        results[corpus] = {
            "subsets": metrics,
            "response_null_unseen": {
                "mean": float(values.mean()),
                "p95": float(np.quantile(values, .95)),
                "maximum": float(values.max()),
                "replicates": len(values),
            },
            "seen_fraction": float(row["seen"].mean()),
        }

    replication_error = max(abs(
        results[corpus]["subsets"]["all"]["embedding_full"]
        - prior["results"][corpus]["embedding_ridge"]["operator_fidelity"])
        for corpus in results)
    hypotheses = {
        "E0_full_embedding_router_replicates": replication_error <= 1e-5,
        "E1_embedding_geometry_generalizes_unseen_tokens": all(
            row["subsets"]["unseen"]["embedding_full"]
            >= row["subsets"]["unseen"]["coarse"] + .02 for row in results.values()),
        "E2_embedding_geometry_beats_response_null": all(
            row["subsets"]["unseen"]["embedding_full"]
            >= row["response_null_unseen"]["p95"] + .03 for row in results.values()),
        "E3_context_span_contains_a_usable_lexical_router": all(
            row["subsets"]["all"]["embedding_context_span"] >= .75 for row in results.values()),
        "E4_embedding_router_is_output_low_rank": all(
            row["subsets"]["all"]["embedding_rank_8"]
            >= row["subsets"]["all"]["embedding_full"] - .02 for row in results.values()),
        "E5_embedding_and_context_input_subspaces_align": alignment["mean_canonical_cosine"] >= .50,
        "E6_unseen_identity_cost_is_bounded": all(
            row["subsets"]["unseen"]["embedding_full"]
            >= row["subsets"]["seen"]["embedding_full"] - .15 for row in results.values()),
        "E7_geometry_beats_lookup_on_unseen_identities": all(
            row["subsets"]["unseen"]["embedding_full"]
            >= row["subsets"]["unseen"]["token_table"] + .05 for row in results.values()),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "replication_error": replication_error,
        "alignment": alignment,
        "map_singular_values": [float(value) for value in singular.cpu()],
        "ridge_penalties": {
            "full": full_prepared["penalty"],
            "context_span": restricted_prepared["penalty"],
        },
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Unseen-identity red team for the Layer-17 embedding router", "",
        "The earlier aggregate embedding result is split by whether the exact token identity occurred in discovery. Matched response permutations preserve coarse class and absolute position while destroying lexical pairing.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Prior-run replication error: `{replication_error:.2e}`. Mean canonical cosine between the fitted embedding-input span and tensor context span: `{alignment['mean_canonical_cosine']:.3f}`.",
              "", "## Seen versus unseen token identities", "",
              "| Corpus/subset | N | mean | coarse | token table | full embedding | context-span embedding | rank-4 / rank-8 | null p95 (unseen) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for subset in ("all", "seen", "unseen"):
            metric = row["subsets"][subset]
            lines.append(
                f"| {corpus}/{subset} | {metric['examples']} | {metric['mean']:.3f} | "
                f"{metric['coarse']:.3f} | {metric['token_table']:.3f} | "
                f"{metric['embedding_full']:.3f} | {metric['embedding_context_span']:.3f} | "
                f"{metric['embedding_rank_4']:.3f}/{metric['embedding_rank_8']:.3f} | "
                f"{row['response_null_unseen']['p95']:.3f} |")
    lines += ["", "## Interpretation", "",
              "A high unseen-token score that beats the matched response null identifies shared geometry, not merely recurring token IDs. The restricted context-span map asks a stronger architectural question: whether the same residual directions that parameterize the tensor context also read lexical identity. Rank truncation concerns the fitted router map, not the exact native operator or the whole model."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "replication_error": replication_error,
        "alignment": alignment,
        "summary": {corpus: {
            "seen_fraction": row["seen_fraction"],
            "seen_embedding": row["subsets"]["seen"]["embedding_full"],
            "unseen_coarse": row["subsets"]["unseen"]["coarse"],
            "unseen_lookup": row["subsets"]["unseen"]["token_table"],
            "unseen_embedding": row["subsets"]["unseen"]["embedding_full"],
            "unseen_null_p95": row["response_null_unseen"]["p95"],
            "context_span": row["subsets"]["all"]["embedding_context_span"],
            "rank8": row["subsets"]["all"]["embedding_rank_8"],
        } for corpus, row in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
