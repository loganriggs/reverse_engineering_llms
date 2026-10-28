#!/usr/bin/env python3
"""Remove a declared 50-feature ontology and discover replicated MLP-write residuals."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_operator_probe_convergence import linear_sum_assignment_maximum
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_known_ontology_residual_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_known_ontology_residual_atlas.json"
ARRAYS = TENSOR_ROOT / "bilin18_known_ontology_residual_atlas.npz"
REPORT = TENSOR_ROOT / "BILIN18_KNOWN_ONTOLOGY_RESIDUAL_ATLAS.md"


@torch.inference_mode()
def collect_panel(model, corpus, global_start, positions, layers, batch_size):
    dimension = model.config.n_embd; selected = torch.as_tensor(positions, device="cuda"); wanted = set(layers)
    writes = {layer: [] for layer in layers}; metadata = {key: [] for key in ("global_row", "position", "current_id", "next_id")}
    for start in range(0, len(corpus), batch_size):
        raw = np.asarray(corpus[start:start + batch_size, :129], dtype=np.int64); ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; bus = None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0; details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(normalized) * block.mlp.Right(normalized); write = block.mlp.Down(product)
            if layer in wanted:
                selected_write = write[:, selected].flatten(0, 1).float()
                if not torch.isfinite(selected_write).all():
                    raise FloatingPointError(f"nonfinite MLP write at layer {layer}")
                writes[layer].append(selected_write.cpu().numpy())
            x = pre + write + block.mlp.Down_bias
        batch = len(raw); metadata["global_row"].extend(np.repeat(global_start + start + np.arange(batch), len(positions)).tolist()); metadata["position"].extend(np.tile(positions, batch).tolist()); metadata["current_id"].extend(raw[:, positions].reshape(-1).tolist()); metadata["next_id"].extend(raw[:, np.asarray(positions) + 1].reshape(-1).tolist())
        if (start + batch_size) % 64 == 0: print(f"known-ontology panel={global_start} rows={min(start + batch_size, len(corpus))}/{len(corpus)}", flush=True)
    return {layer: np.concatenate(rows).astype(np.float32) for layer, rows in writes.items()}, {key: np.asarray(value) for key, value in metadata.items()}


def design_matrices(metadata, flags, positions):
    current = flags[metadata["current_id"]].astype(np.float64); following = flags[metadata["next_id"]].astype(np.float64)
    position = np.zeros((len(current), len(positions)), dtype=np.float64); position[:, :] = np.asarray(metadata["position"])[:, None] == np.asarray(positions)[None]
    return {"current50": current, "next50": following, "declared": np.concatenate((current, following, position), axis=1)}


def standardize(train, test):
    mean = train.mean(0); scale = train.std(0); keep = scale > 1e-8
    return (train[:, keep] - mean[keep]) / scale[keep], (test[:, keep] - mean[keep]) / scale[keep]


def multivariate_r2(target, prediction, mean):
    return float(1 - np.square(target - prediction).sum() / max(np.square(target - mean).sum(), 1e-30))


def ridge_fit(train_x, train_y, test_x, alphas, groups):
    train_x, test_x = standardize(np.asarray(train_x, dtype=np.float64), np.asarray(test_x, dtype=np.float64)); train_y = np.asarray(train_y, dtype=np.float64); test_x = np.asarray(test_x)
    ymean = train_y.mean(0); centered = train_y - ymean
    validation = np.asarray(groups) % 4 == 0; fitting = ~validation; best = None
    for alpha in alphas:
        gram = train_x[fitting].T @ train_x[fitting] + alpha * np.eye(train_x.shape[1]); coefficient = np.linalg.solve(gram, train_x[fitting].T @ centered[fitting]); prediction = ymean + train_x[validation] @ coefficient
        score = multivariate_r2(train_y[validation], prediction, ymean)
        if best is None or score > best[0]: best = score, alpha
    alpha = best[1]; coefficient = np.linalg.solve(train_x.T @ train_x + alpha * np.eye(train_x.shape[1]), train_x.T @ centered)
    return ymean + train_x @ coefficient, ymean + test_x @ coefficient, float(alpha), float(best[0])


def token_lookup(train_ids, train_y, test_ids, pseudocount):
    vocab = int(max(np.max(train_ids), np.max(test_ids))) + 1; mean = train_y.mean(0); counts = np.bincount(train_ids, minlength=vocab).astype(np.float64); sums = np.zeros((vocab, train_y.shape[1]), dtype=np.float64); np.add.at(sums, train_ids, train_y)
    denominator = counts[:, None] + pseudocount; table = np.divide(sums + pseudocount * mean, denominator, out=np.broadcast_to(mean, sums.shape).copy(), where=denominator > 0); return table[test_ids], float(np.mean(counts[test_ids] > 0))


def pca(values, rank):
    values = np.asarray(values, dtype=np.float64); mean = values.mean(0); centered = values - mean; covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T)); order = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[order]; basis = eigenvectors[:, order[:rank]].T
    return mean, basis, eigenvalues


def project(values, mean, basis):
    return mean + ((values - mean) @ basis.T) @ basis


def subspace_overlap(first, second):
    return float(np.linalg.svd(np.asarray(first) @ np.asarray(second).T, compute_uv=False).mean())


def match_axes(first, second):
    score = np.asarray(first) @ np.asarray(second).T; rows, columns = linear_sum_assignment_maximum(np.abs(score)); order = np.argsort(rows); columns = columns[order]; values = score[np.arange(len(first)), columns]
    return columns, values


def best_feature(scores, metadata, flags):
    candidates = []
    for role, ids in (("current", metadata["current_id"]), ("next", metadata["next_id"])):
        labels = flags[ids]
        for index, name in enumerate(FEATURES):
            if labels[:, index].sum() < 10 or labels[:, index].sum() > len(labels) - 10: continue
            value = float(auc(labels[:, index], scores)); oriented = max(value, 1 - value); candidates.append({"feature": f"{role}:{name}", "auroc": oriented, "positive_pole": value >= .5})
    return max(candidates, key=lambda row: row["auroc"])


def clean(text):
    return text.replace("\n", "\\n").replace("\t", "\\t")


def examples(tokenizer, corpus, metadata, scores, count, before, after):
    output = {}
    for pole, order in (("positive", np.argsort(-scores)), ("negative", np.argsort(scores))):
        rows = []
        for index in order[:count]:
            global_row = int(metadata["global_row"][index]); position = int(metadata["position"][index]); tokens = np.asarray(corpus[global_row], dtype=np.int64); left = max(0, position - before); right = min(len(tokens), position + after + 1)
            rows.append({"global_row": global_row, "position": position, "score": float(scores[index]), "current": clean(tokenizer.decode([int(tokens[position])])), "next": clean(tokenizer.decode([int(tokens[position + 1])])), "context": clean(tokenizer.decode(tokens[left:right].tolist()))})
        output[pole] = rows
    return output


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    corpus = np.load(config["corpus"], mmap_mode="r"); model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    panels = {}
    for name, offset in (("discovery", config["discovery_start"]), ("replication", config["replication_start"])):
        values, metadata = collect_panel(model, corpus[offset:offset + config["rows_per_panel"]], offset, config["positions"], config["layers"], config["batch_size"]); panels[name] = {"values": values, "metadata": metadata, "design": design_matrices(metadata, flags, config["positions"])}
    results = {}; candidate_pool = []; arrays = {}
    for layer in config["layers"]:
        discovery = panels["discovery"]["values"][layer]; replication = panels["replication"]["values"][layer]; dmeta = panels["discovery"]["metadata"]; rmeta = panels["replication"]["metadata"]
        dmean = discovery.mean(0); layer_row = {"designs": {}}
        predictions = {}
        for name in ("current50", "next50", "declared"):
            dpred, rpred, alpha, validation = ridge_fit(panels["discovery"]["design"][name], discovery, panels["replication"]["design"][name], config["ridge_alphas"], dmeta["global_row"])
            predictions[name] = (dpred, rpred); layer_row["designs"][name] = {"alpha": alpha, "discovery_validation_r2": validation, "replication_r2": multivariate_r2(replication, rpred, dmean)}
        lookup, coverage = token_lookup(dmeta["current_id"], discovery, rmeta["current_id"], config["token_lookup_pseudocount"]); layer_row["current_token_lookup"] = {"replication_r2": multivariate_r2(replication, lookup, dmean), "coverage": coverage}
        _, natural_basis, _ = pca(discovery, max(config["pca_ranks"])); layer_row["natural_pca"] = {str(rank): multivariate_r2(replication, project(replication, dmean, natural_basis[:rank]), dmean) for rank in config["pca_ranks"]}
        dpred, rpred = predictions["declared"]; dresidual = discovery - dpred; rresidual = replication - rpred
        drmean, dbasis, deigen = pca(dresidual, max(config["pca_ranks"])); rrmean, rbasis, reigen = pca(rresidual, max(config["pca_ranks"])); layer_row["residual"] = {}
        for rank in config["pca_ranks"]:
            combined = rpred + project(rresidual, drmean, dbasis[:rank]); layer_row["residual"][str(rank)] = {"subspace_overlap": subspace_overlap(dbasis[:rank], rbasis[:rank]), "combined_total_r2": multivariate_r2(replication, combined, dmean), "residual_fidelity": multivariate_r2(rresidual, project(rresidual, drmean, dbasis[:rank]), drmean)}
        rank = config["candidate_rank"]; permutation, axis_cosines = match_axes(dbasis[:rank], rbasis[:rank]); total_eigen = max(deigen.sum(), 1e-30)
        layer_row["axis_cosines"] = axis_cosines.tolist(); layer_row["residual_eigen_fraction"] = (deigen[:rank] / total_eigen).tolist()
        for axis in range(rank):
            scores = (rresidual - drmean) @ dbasis[axis]; feature = best_feature(scores, rmeta, flags); candidate_pool.append({"layer": layer, "axis": axis, "replication_axis": int(permutation[axis]), "axis_cosine": float(axis_cosines[axis]), "eigen_fraction": float(deigen[axis] / total_eigen), "best_declared_feature": feature, "_scores": scores})
        arrays[f"residual_basis_L{layer}"] = dbasis.astype(np.float32); arrays[f"residual_eigenvalues_L{layer}"] = deigen.astype(np.float32)
        results[str(layer)] = layer_row; print(f"known ontology residual L{layer}", flush=True)
    eligible = [row for row in candidate_pool if abs(row["axis_cosine"]) >= config["minimum_axis_cosine"]]; eligible.sort(key=lambda row: abs(row["axis_cosine"]) * row["eigen_fraction"], reverse=True); selected = eligible[:config["maximum_candidates"]]
    for row in selected:
        scores = row.pop("_scores"); row["examples"] = examples(tokenizer, corpus, panels["replication"]["metadata"], scores, config["examples_per_pole"], config["context_before"], config["context_after"])
    for row in candidate_pool: row.pop("_scores", None)
    substantial = sum(results[str(layer)]["designs"]["declared"]["replication_r2"] >= .25 for layer in config["layers"]); lookup_incomplete = all(results[str(layer)]["current_token_lookup"]["replication_r2"] < .75 for layer in config["layers"]); recurring = sum(results[str(layer)]["residual"]["16"]["subspace_overlap"] >= .70 for layer in config["layers"]); additive = sum(results[str(layer)]["residual"]["64"]["combined_total_r2"] - results[str(layer)]["designs"]["declared"]["replication_r2"] >= .20 for layer in config["layers"]); escaped = sum(row["best_declared_feature"]["auroc"] < .75 for row in selected)
    hypotheses = {
        "K0_panels_are_fresh_and_task_blind_before_residualization": {"supported": config["discovery_start"] + config["rows_per_panel"] <= config["replication_start"], "ranges": [[config["discovery_start"], config["discovery_start"] + config["rows_per_panel"] - 1], [config["replication_start"], config["replication_start"] + config["rows_per_panel"] - 1]]},
        "K1_declared_features_explain_substantial_write_variance": {"supported": substantial >= 3, "passing": substantial},
        "K2_exact_current_token_is_not_a_complete_ontology": {"supported": lookup_incomplete, "r2": {str(layer): results[str(layer)]["current_token_lookup"]["replication_r2"] for layer in config["layers"]}},
        "K3_residual_subspaces_recur": {"supported": recurring >= 3, "passing": recurring},
        "K4_residual_coordinates_add_material_coverage": {"supported": additive == len(config["layers"]), "passing": additive},
        "K5_stable_residual_axes_exist": {"supported": len(eligible) >= 4, "eligible": len(eligible)},
        "K6_residual_axes_escape_the_declared_vocabulary": {"supported": bool(selected) and escaped >= len(selected) / 2, "passing": escaped, "selected": len(selected)},
        "K7_late_known_plus_residual_code_is_compact": {"supported": results["17"]["residual"]["16"]["combined_total_r2"] >= .90, "r2": results["17"]["residual"]["16"]["combined_total_r2"]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    np.savez_compressed(ARRAYS, **arrays); OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "selected_candidates": selected}, indent=2, allow_nan=False))
    lines = ["# Known-ontology residual atlas", "", "A declared bank of fifty token features is used only after collecting random FineWeb MLP writes. Linear models predict each layer's exact write from current-token features, next-token features, or both plus position. PCA is then fit to the unexplained write residual on discovery and evaluated on a disjoint panel. Exact current-token lookup is a high-capacity lexical control.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Current50 | Next50 | Declared+position | Token lookup | Natural PCA16/64 | Residual overlap16/64 | Known+residual total R2 16/64 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; lines.append(f"| {layer} | {row['designs']['current50']['replication_r2']:.3f} | {row['designs']['next50']['replication_r2']:.3f} | {row['designs']['declared']['replication_r2']:.3f} | {row['current_token_lookup']['replication_r2']:.3f} | {row['natural_pca']['16']:.3f}/{row['natural_pca']['64']:.3f} | {row['residual']['16']['subspace_overlap']:.3f}/{row['residual']['64']['subspace_overlap']:.3f} | {row['residual']['16']['combined_total_r2']:.3f}/{row['residual']['64']['combined_total_r2']:.3f} |")
    lines += ["", "## Frozen residual-axis candidates", "", "| Candidate | Axis cosine | Residual variance | Best declared feature | AUROC |", "|---|---:|---:|---|---:|"]
    for row in selected: lines.append(f"| L{row['layer']}A{row['axis']} | {row['axis_cosine']:+.3f} | {row['eigen_fraction']:.3f} | {row['best_declared_feature']['feature']} | {row['best_declared_feature']['auroc']:.3f} |")
    lines += ["", "## Guardrails", "", "The fifty labels are an anchor ontology, not exhaustive ground truth. Next-token labels are diagnostic oracle metadata and do not define an executable forward router. Residual PCA axes are candidate coordinates: subspace recurrence is stronger evidence than individual-axis recurrence, and semantic names require held-out causal transfer. High token-lookup R2 would indicate lexical inheritance, not a complete explanation of contextual use."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
