#!/usr/bin/env python3
"""Finite suffix validation of known semantic and residual MLP-write codes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_known_ontology_residual_atlas import collect_panel, design_matrices, multivariate_r2, pca, project
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from redteam_bilin18_recursive_feature_ontology import interaction_design, vocabulary_classes
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_taskblind_mlp_subspace_finite import intervene_write, random_basis, target_product_and_write
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric
from transformers import AutoTokenizer


CONFIG = Path("configs/bilin18_known_ontology_residual_finite.json")
OUTPUT = TENSOR_ROOT / "bilin18_known_ontology_residual_finite.json"
REPORT = TENSOR_ROOT / "BILIN18_KNOWN_ONTOLOGY_RESIDUAL_FINITE.md"


def fit_fixed_ridge(train_x, train_y, alpha):
    train_x = np.asarray(train_x, dtype=np.float64); train_y = np.asarray(train_y, dtype=np.float64); mean = train_x.mean(0); scale = train_x.std(0); keep = scale > 1e-8; standardized = (train_x[:, keep] - mean[keep]) / scale[keep]; ymean = train_y.mean(0); centered = train_y - ymean
    coefficient = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(standardized.shape[1]), standardized.T @ centered)
    return {"mean": mean, "scale": scale, "keep": keep, "ymean": ymean, "coefficient": coefficient}


def ridge_predict(model, values):
    standardized = (np.asarray(values, dtype=np.float64)[:, model["keep"]] - model["mean"][model["keep"]]) / model["scale"][model["keep"]]
    return model["ymean"] + standardized @ model["coefficient"]


def make_lookup(metadata):
    return {(int(row), int(position)): index for index, (row, position) in enumerate(zip(metadata["global_row"], metadata["position"]))}


def main():
    config = json.loads(CONFIG.read_text()); known = json.loads(Path(config["known_source"]).read_text()); recursive = json.loads(Path(config["recursive_source"]).read_text()); rng = np.random.default_rng(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); classes = vocabulary_classes(tokenizer, flags); corpus = np.load(config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    panels = {}
    for name, offset, rows, positions in (("discovery", config["discovery_start"], config["discovery_rows"], config["discovery_positions"]), ("validation", config["validation_start"], config["validation_rows"], config["validation_positions"])):
        values, metadata = collect_panel(model, corpus[offset:offset + rows], offset, positions, config["layers"], config["collection_batch_size"]); base = design_matrices(metadata, flags, config["discovery_positions"])["declared"]; interaction, _ = interaction_design(metadata, classes); panels[name] = {"values": values, "metadata": metadata, "base": base, "extended": np.concatenate((base, interaction), axis=1)}
    prepared = {}; observational = {}
    for layer in config["layers"]:
        discovery = panels["discovery"]["values"][layer]; validation = panels["validation"]["values"][layer]; dmean = discovery.mean(0)
        base_model = fit_fixed_ridge(panels["discovery"]["base"], discovery, known["results"][str(layer)]["designs"]["declared"]["alpha"]); extended_model = fit_fixed_ridge(panels["discovery"]["extended"], discovery, recursive["results"][str(layer)]["extended_alpha"])
        base_d = ridge_predict(base_model, panels["discovery"]["base"]); base_v = ridge_predict(base_model, panels["validation"]["base"]); extended_d = ridge_predict(extended_model, panels["discovery"]["extended"]); extended_v = ridge_predict(extended_model, panels["validation"]["extended"])
        base_res_d = discovery - base_d; base_res_v = validation - base_v; ext_res_d = discovery - extended_d; ext_res_v = validation - extended_v
        base_mean, base_basis, _ = pca(base_res_d, max(config["ranks"])); ext_mean, ext_basis, _ = pca(ext_res_d, max(config["ranks"])); natural_mean, natural_basis, _ = pca(discovery, max(config["ranks"]))
        approximation = {"semantic": base_v, "recursive_semantic": extended_v}
        for rank in config["ranks"]:
            approximation[f"semantic_residual_{rank}"] = base_v + project(base_res_v, base_mean, base_basis[:rank]); approximation[f"recursive_residual_{rank}"] = extended_v + project(ext_res_v, ext_mean, ext_basis[:rank]); approximation[f"natural_pca_{rank}"] = project(validation, natural_mean, natural_basis[:rank])
        random = []
        for _ in range(config["random_replicates"]):
            basis = random_basis(validation.shape[1], 64, rng); random.append(base_v + project(base_res_v, base_mean, basis))
        approximation.update({f"random_residual_64_{index}": value for index, value in enumerate(random)})
        prepared[layer] = {name: torch.as_tensor(value, device="cuda", dtype=torch.float32) for name, value in approximation.items()}; prepared[layer]["lookup"] = make_lookup(panels["validation"]["metadata"])
        observational[str(layer)] = {name: multivariate_r2(validation, value, dmean) for name, value in approximation.items()}
    methods = ["native", "deleted", "semantic", "recursive_semantic"] + [f"{prefix}_{rank}" for prefix in ("semantic_residual", "recursive_residual", "natural_pca") for rank in config["ranks"]] + [f"random_residual_64_{index}" for index in range(config["random_replicates"])]
    metrics = {layer: {name: empty_metric() for name in methods} for layer in config["layers"]}; maximum_error = 0.0; validation_corpus = corpus[config["validation_start"]:config["validation_start"] + config["validation_rows"]]
    for position in config["validation_positions"]:
        length = position + 1
        for local_row in range(len(validation_corpus)):
            raw = np.asarray(validation_corpus[local_row:local_row + 1, :length + 1], dtype=np.int64); ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda"); reference = full_logits(model, ids); global_row = config["validation_start"] + local_row
            for layer in config["layers"]:
                _, native_write = target_product_and_write(model, ids, layer); index = prepared[layer]["lookup"][(global_row, position)]; conditions = {"native": native_write, "deleted": torch.zeros_like(native_write)}
                for name in methods[2:]: conditions[name] = prepared[layer][name][index:index + 1]
                names, logits = intervene_write(model, ids, layer, conditions); outputs = {name: logits[i] for i, name in enumerate(names)}; maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], outputs["native"], outputs["deleted"], outputs[name], targets)
        print(f"known ontology finite position={position}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}; residual_gain = []; recursive_gain = []; random_wins = []; equivalent = []; observations = []; finite = []
    evaluated = ["semantic", "recursive_semantic"] + [f"{prefix}_{rank}" for prefix in ("semantic_residual", "recursive_residual", "natural_pca") for rank in config["ranks"]]
    for layer in config["layers"]:
        row = results[str(layer)]; residual_gain.append(row["semantic_residual_64"]["causal_fidelity"] - row["semantic"]["causal_fidelity"] >= .15); recursive_gain.append(row["recursive_semantic"]["causal_fidelity"] - row["semantic"]["causal_fidelity"] >= .03); p95 = float(np.quantile([row[f"random_residual_64_{index}"]["causal_fidelity"] for index in range(config["random_replicates"])], .95)); row["random_residual_64_p95"] = p95; random_wins.append(row["semantic_residual_64"]["causal_fidelity"] > p95); equivalent.append(abs(row["natural_pca_64"]["causal_fidelity"] - row["semantic_residual_64"]["causal_fidelity"]) < .10)
        observations.extend(observational[str(layer)][name] for name in evaluated); finite.extend(row[name]["causal_fidelity"] for name in evaluated)
    correlation = float(np.corrcoef(observations, finite)[0, 1])
    hypotheses = {
        "C0_native_replay_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "C1_declared_semantics_have_late_causal_value": {"supported": results["17"]["semantic"]["causal_fidelity"] >= .25, "fidelity": results["17"]["semantic"]["causal_fidelity"]},
        "C2_residual_coordinates_add_finite_value": {"supported": all(residual_gain), "passing": int(sum(residual_gain))},
        "C3_recursive_semantics_add_finite_value": {"supported": sum(recursive_gain) >= 2, "passing": int(sum(recursive_gain))},
        "C4_late_semantic_plus_residual_is_compact": {"supported": results["17"]["semantic_residual_16"]["causal_fidelity"] >= .90, "fidelity": results["17"]["semantic_residual_16"]["causal_fidelity"]},
        "C5_residual_coordinates_beat_random": {"supported": all(random_wins), "passing": int(sum(random_wins))},
        "C6_natural_and_semantic_residual_codes_are_equivalent": {"supported": all(equivalent), "passing": int(sum(equivalent))},
        "C7_write_reconstruction_predicts_finite_behavior": {"supported": correlation >= .70, "correlation": correlation},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "observational_write_r2": observational, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causality of known semantic and residual MLP codes", "", "On a fourth FineWeb panel, the destination MLP write is replaced by: the fifty-feature semantic prediction, the recursive coarse-interaction prediction, either prediction plus rank-16/64 residual coordinates, or ordinary natural-write PCA. The complete nonlinear suffix recomputes. Because next-token labels enter the semantic prediction, it is a hindsight explanatory code rather than an executable model-internal router.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Semantic | Recursive | Semantic+res16/64 | Recursive+res16/64 | Natural PCA16/64 | Random res64 p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; lines.append(f"| {layer} | {row['semantic']['causal_fidelity']:.3f} | {row['recursive_semantic']['causal_fidelity']:.3f} | {row['semantic_residual_16']['causal_fidelity']:.3f}/{row['semantic_residual_64']['causal_fidelity']:.3f} | {row['recursive_residual_16']['causal_fidelity']:.3f}/{row['recursive_residual_64']['causal_fidelity']:.3f} | {row['natural_pca_16']['causal_fidelity']:.3f}/{row['natural_pca_64']['causal_fidelity']:.3f} | {row['random_residual_64_p95']:.3f} |")
    lines += ["", "## Guardrails", "", "These are sufficiency tests relative to deleting one MLP write. Semantic predictions use the true next token and therefore measure explanatory compression, not a deployable forward computation. Residual projections observe the native write to encode its coordinates. Similar fidelity between natural PCA and semantic-plus-residual codes establishes behavioral equivalence at this rank, not identical semantic axes."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
