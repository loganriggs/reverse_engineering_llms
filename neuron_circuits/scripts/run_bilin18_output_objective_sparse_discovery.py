#!/usr/bin/env python3
"""Label-free sparse rotations of the stable output tensor core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_router_jacobian_sparse_basis_redteam import (
    fast_ica, greedy_overlap, varimax,
)
from run_bilin18_layer17_tangent_output_semantics import (
    collect, evaluate_individual, select_individual, vocabulary_table,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_output_objective_sparse_discovery.json")
OUTPUT = TENSOR_ROOT / "bilin18_output_objective_sparse_discovery.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_OBJECTIVE_SPARSE_DISCOVERY.md"


def standardized(scores, mean=None, scale=None):
    if mean is None: mean = scores.mean(0)
    if scale is None: scale = scores.std(0).clamp_min(1e-6)
    return (scores - mean) / scale, mean, scale


def absolute_excess_kurtosis(scores):
    z, _, _ = standardized(scores)
    return float((z.pow(4).mean(0) - 3).abs().mean())


def fit_sparse_bases(scores, generator, config):
    z, _, _ = standardized(scores)
    runs = [fast_ica(z, generator, config["ica_iterations"]) for _ in range(config["ica_starts"])]
    winner = int(np.argmax([metadata["absolute_excess_kurtosis"] for _, metadata in runs]))
    ica_q, ica_metadata = runs[winner]
    overlaps = [greedy_overlap(ica_q, q) for index, (q, _) in enumerate(runs) if index != winner]
    varimax_q, varimax_metadata = varimax(z)
    return {"ica": ica_q, "varimax": varimax_q}, {
        "ica": {**ica_metadata, "selected_start": winner, "cross_start_overlap": overlaps},
        "varimax": varimax_metadata,
    }


def semantic_result(train_scores, test_scores, train_labels, test_labels, names, minimum):
    selection = select_individual(train_scores.numpy(), train_labels, names, minimum)
    result = evaluate_individual(test_scores.numpy(), test_labels, selection)
    result["distinct_labels"] = len({row["label"] for row in selection})
    return selection, result


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts, vocabulary, vocabulary_names = vocabulary_table(tokenizer)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    native_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    weight_scores = (model.lm_head.weight.float()[:tokenizer.vocab_size] @ native_axes).cpu()
    token_index = np.arange(tokenizer.vocab_size)
    fit_mask = token_index % 2 == 0; test_mask = ~fit_mask
    generator = torch.Generator().manual_seed(config["seed"])
    sparse, optimization = fit_sparse_bases(weight_scores[fit_mask], generator, config)
    bases = {"native": torch.eye(config["rank"]), **sparse}
    random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(config["random_rotations"])]

    weight_results, weight_selections = {}, {}
    for name, q in bases.items():
        selection, result = semantic_result(
            weight_scores[fit_mask] @ q, weight_scores[test_mask] @ q,
            vocabulary[fit_mask], vocabulary[test_mask], vocabulary_names,
            config["minimum_positive"],
        )
        result["heldout_absolute_excess_kurtosis"] = absolute_excess_kurtosis(weight_scores[test_mask] @ q)
        weight_selections[name] = selection; weight_results[name] = result
    random_semantics, random_kurtosis = [], []
    for q in random_qs:
        _, result = semantic_result(
            weight_scores[fit_mask] @ q, weight_scores[test_mask] @ q,
            vocabulary[fit_mask], vocabulary[test_mask], vocabulary_names,
            config["minimum_positive"],
        )
        random_semantics.append(result["median_auc"])
        random_kurtosis.append(absolute_excess_kurtosis(weight_scores[test_mask] @ q))

    # Stability is measured without labels: refit on two disjoint quarters of
    # the vocabulary, then compare the recovered physical coordinates.
    stability = {}
    quarter_bases = []
    for residue in (0, 2):
        local_generator = torch.Generator().manual_seed(config["seed"] + residue + 1)
        fitted, _ = fit_sparse_bases(weight_scores[token_index % 4 == residue], local_generator, config)
        quarter_bases.append(fitted)
    for name in ("ica", "varimax"):
        stability[name] = greedy_overlap(quarter_bases[0][name], quarter_bases[1][name])

    collect_config = {**config, "discovery": {"path": config["corpus"], "start": config["discovery_start"], "rows": config["rows"]}}
    print("collect natural discovery", flush=True)
    discovery = collect(model, tokenizer, collect_config["discovery"], collect_config, native_axes, vocabulary, vocabulary_names)
    evaluation_spec = {"path": config["corpus"], "start": config["evaluation_start"], "rows": config["rows"]}
    print("collect natural evaluation", flush=True)
    evaluation = collect(model, tokenizer, evaluation_spec, collect_config, native_axes, vocabulary, vocabulary_names)
    natural_results, natural_selections = {}, {}
    for basis_name, q in bases.items():
        natural_results[basis_name] = {}; natural_selections[basis_name] = {}
        for component in ("target", "top1"):
            selection, result = semantic_result(
                discovery["scores"][component] @ q,
                evaluation["scores"][component] @ q,
                discovery["labels"], evaluation["labels"], discovery["label_names"],
                config["minimum_positive"],
            )
            natural_selections[basis_name][component] = selection
            natural_results[basis_name][component] = result
    random_natural = {"target": [], "top1": []}
    for q in random_qs:
        for component in ("target", "top1"):
            _, result = semantic_result(
                discovery["scores"][component] @ q,
                evaluation["scores"][component] @ q,
                discovery["labels"], evaluation["labels"], discovery["label_names"],
                config["minimum_positive"],
            )
            random_natural[component].append(result["median_auc"])

    sparse_name = max(("ica", "varimax"), key=lambda name: weight_results[name]["heldout_absolute_excess_kurtosis"])
    random_semantic_p95 = float(np.quantile(random_semantics, .95))
    random_kurtosis_p95 = float(np.quantile(random_kurtosis, .95))
    natural_sparse_median = min(natural_results[sparse_name][component]["median_auc"] for component in ("target", "top1"))
    hypotheses = {
        "O0_basis_fit_is_label_free": {"supported": True},
        "O1_sparse_objective_improves_kurtosis": {"supported": weight_results[sparse_name]["heldout_absolute_excess_kurtosis"] > random_kurtosis_p95},
        "O2_sparse_objective_improves_weight_semantics": {"supported": weight_results[sparse_name]["median_auc"] >= weight_results["native"]["median_auc"] + .03},
        "O3_sparse_objective_is_semantically_privileged": {"supported": weight_results[sparse_name]["median_auc"] >= random_semantic_p95 + .03},
        "O4_sparse_coordinates_are_reproducible": {"supported": stability[sparse_name] >= .80},
        "O5_discovered_objectives_are_diverse": {"supported": weight_results[sparse_name]["distinct_labels"] >= 5},
        "O6_natural_semantics_transfer": {"supported": natural_sparse_median >= .70},
    }
    result = {
        "config": config, "hypotheses": hypotheses, "selected_sparse_basis": sparse_name,
        "optimization": optimization, "split_stability": stability,
        "random": {
            "weight_semantic_p95": random_semantic_p95,
            "weight_kurtosis_p95": random_kurtosis_p95,
            "natural_target_p95": float(np.quantile(random_natural["target"], .95)),
            "natural_top1_p95": float(np.quantile(random_natural["top1"], .95)),
        },
        "weight_results": weight_results, "weight_selections": weight_selections,
        "natural_results": natural_results, "natural_selections": natural_selections,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = [
        "# Label-free discovery of output objectives", "",
        "ICA and varimax rotate the stable rank-16 output tensor core using only its weight-derived vocabulary loadings. Human-readable token predicates are consulted only after the bases are frozen. Native HOSVD axes and matched random rotations of the identical physical subspace are controls.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Selected label-free sparse basis: **{sparse_name}**.", "", "| Basis | Weight semantic AUROC | Weight kurtosis | Split stability | Natural target/top1 AUROC | Distinct labels |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("native", "ica", "varimax"):
        stability_text = "—" if name == "native" else f"{stability[name]:.3f}"
        lines.append(f"| {name} | {weight_results[name]['median_auc']:.3f} | {weight_results[name]['heldout_absolute_excess_kurtosis']:.3f} | {stability_text} | {natural_results[name]['target']['median_auc']:.3f}/{natural_results[name]['top1']['median_auc']:.3f} | {weight_results[name]['distinct_labels']} |")
    lines.append(f"| random rotation p95 | {random_semantic_p95:.3f} | {random_kurtosis_p95:.3f} | — | {result['random']['natural_target_p95']:.3f}/{result['random']['natural_top1_p95']:.3f} | — |")
    lines += ["", "## Discovered weight-side labels", "", "| Axis | ICA | Varimax |", "|---:|---|---|"]
    for index in range(config["rank"]):
        left = weight_selections["ica"][index]; right = weight_selections["varimax"][index]
        lines.append(f"| {index} | `{left['label']}` ({weight_results['ica']['mode_aucs'][index]:.2f}) | `{right['label']}` ({weight_results['varimax']['mode_aucs'][index]:.2f}) |")
    lines += ["", "## Guardrail", "", "A sparse rotation is a candidate coordinate system inside an already discovered physical subspace. High semantic AUROC is not enough: it must beat matched rotations and reproduce from disjoint weight samples before individual axes count as model-preferred objectives. If only the whole subspace passes, the correct ontology remains a semantic output subspace with non-identifiable internal coordinates."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "selected_sparse_basis": sparse_name, "split_stability": stability, "weight_results": weight_results, "natural_results": natural_results, "random": result["random"]}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
