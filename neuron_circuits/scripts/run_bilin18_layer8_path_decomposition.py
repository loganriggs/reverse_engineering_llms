#!/usr/bin/env python3
"""Split routed layer-8 edits into direct and attention-mediated layer-9 paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    prefix_to_layer,
    repeat_batch,
    route_indices,
    select_sites,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_path_decomposition.json")
OUTPUT = TENSOR_ROOT / "layer8_path_decomposition.json"
REPORT = TENSOR_ROOT / "LAYER8_PATH_DECOMPOSITION.md"
FIGURE = TENSOR_ROOT / "layer8_path_decomposition.png"
PATHS = ("direct", "attention", "additive", "full")


@torch.inference_mode()
def immediate_path_hybrids(block, native_x, changed_x, x0, value_bus):
    """Return native/direct/attention/full outputs of the next block.

    Direct changes the residual input while clamping the attention write.
    Attention changes the attention write while clamping the residual input.
    The MLP is recomputed for every hybrid.
    """
    dimension = native_x.shape[-1]
    mixed_native = block.lambdas[0] * native_x + block.lambdas[1] * x0
    mixed_changed = block.lambdas[0] * changed_x + block.lambdas[1] * x0
    # Run branches separately. Batched squared-attention contractions differ
    # from the checkpoint's batch-1 native path by ~1e-3 in float32, which is
    # large enough to contaminate a path-recomposition invariant.
    attention_native, returned_bus = block.attn(
        F.rms_norm(mixed_native, (dimension,)), value_bus,
    )
    attention_changed, changed_bus = block.attn(
        F.rms_norm(mixed_changed, (dimension,)), value_bus,
    )
    if not torch.equal(returned_bus, changed_bus):
        raise AssertionError("the checkpoint value bus changed across layer-9 branches")
    pre = {
        "native": mixed_native + attention_native,
        "direct": mixed_changed + attention_native,
        "attention": mixed_native + attention_changed,
        "full": mixed_changed + attention_changed,
    }
    outputs = {}
    for name, value in pre.items():
        # Preserve the native batch-1 contraction path for the same reason that
        # attention branches are evaluated separately above.
        normalized = F.rms_norm(value, (dimension,))
        write = block.mlp.Down(
            block.mlp.Left(normalized) * block.mlp.Right(normalized)
        ) + block.mlp.Down_bias
        outputs[name] = value + write
    return outputs, returned_bus


@torch.inference_mode()
def suffix_after_layer(model, layer, x0, value_bus, layer_output, position):
    dimension = model.config.n_embd
    x = layer_output
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)


def additive_effect(native, direct, attention):
    return (direct - native) + (attention - native)


def summarize(records, routes):
    within_route = {}
    route_to_oracle = {}
    for route in routes:
        rows = [record["routes"][route] for record in records]
        within_route[route] = {}
        for path in ("direct", "attention", "additive"):
            within_route[route][path] = {
                "median_downstream_fisher_fidelity_to_full": float(np.median([
                    row["paths"][path]["downstream_fisher_fidelity_to_full"] for row in rows
                ])),
                "median_downstream_fisher_cosine_to_full": float(np.median([
                    row["paths"][path]["downstream_fisher_cosine_to_full"] for row in rows
                ])),
                "median_layer9_fidelity_to_full": float(np.median([
                    row["paths"][path]["layer9_fidelity_to_full"] for row in rows
                ])),
            }
        within_route[route]["median_layer9_interaction_energy_fraction"] = float(np.median([
            row["layer9_interaction_energy_fraction"] for row in rows
        ]))
        within_route[route]["median_full_native_kl"] = float(np.median([
            row["paths"]["full"]["native_kl"] for row in rows
        ]))
        for path in ("direct", "attention", "additive"):
            within_route[route][path]["median_native_kl"] = float(np.median([
                row["paths"][path]["native_kl"] for row in rows
            ]))
            within_route[route][path]["median_native_kl_ratio_to_full"] = float(np.median([
                row["paths"][path]["native_kl"]
                / max(row["paths"]["full"]["native_kl"], 1e-30)
                for row in rows
            ]))
        if route != "oracle":
            route_to_oracle[route] = {}
            for path in PATHS:
                route_to_oracle[route][path] = {
                    "median_fisher_fidelity": float(np.median([
                        record["route_to_oracle"][route][path]["fisher_fidelity"]
                        for record in records
                    ])),
                    "median_fisher_cosine": float(np.median([
                        record["route_to_oracle"][route][path]["fisher_cosine"]
                        for record in records
                    ])),
                }
    return {"within_route": within_route, "route_to_oracle": route_to_oracle}


@torch.inference_mode()
def evaluate_corpus(model, config, corpus_config, basis, values, vectors):
    scan_config = {**config, "positions": [config["causal_position"]]}
    states, _ = collect(model, config["layer"], corpus_config, scan_config)
    selected = select_sites(states, basis, values, vectors, config["sites_per_query"])
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    layer = config["layer"]
    next_layer = config["next_layer"]
    position = config["causal_position"]
    routes = config["routes"]
    records = []
    recomposition_errors = []
    for query, query_sites in enumerate(selected):
        for row_index in query_sites:
            raw = np.asarray(rows[row_index, :position + 1], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")[None]
            x0, value_bus, pre8, normalized8 = prefix_to_layer(model, ids, layer)
            block8 = model.blocks[layer]
            native_write8 = block8.mlp.Down(
                block8.mlp.Left(normalized8) * block8.mlp.Right(normalized8)
            ) + block8.mlp.Down_bias
            native_x8 = pre8 + native_write8
            x = normalized8[0, position].float()
            z = x @ basis
            indices = route_indices(
                z, values, vectors, config["budget"], config["coarse_dimension"],
                config["refine_dimension"], config["shortlist"],
            )
            route_outputs = {}
            route_logits = {}
            for route in routes:
                mode_directions = basis @ vectors[query, :, indices[route][query]]
                raw_delta = -(x @ mode_directions) @ mode_directions.T
                delta = raw_delta * (
                    config["edit_norm_fraction"] * x.norm()
                    / raw_delta.norm().clamp_min(1e-20)
                )
                changed_input8 = normalized8.clone()
                changed_input8[0, position] = (x + delta).to(changed_input8.dtype)
                changed_write8 = block8.mlp.Down(
                    block8.mlp.Left(changed_input8) * block8.mlp.Right(changed_input8)
                ) + block8.mlp.Down_bias
                changed_x8 = pre8 + changed_write8
                hybrids, bus9 = immediate_path_hybrids(
                    model.blocks[next_layer], native_x8, changed_x8, x0, value_bus,
                )
                native_forward9, _ = model.blocks[next_layer](native_x8, value_bus, x0)
                changed_forward9, _ = model.blocks[next_layer](changed_x8, value_bus, x0)
                recomposition_errors.extend((
                    float((hybrids["native"] - native_forward9).abs().max()),
                    float((hybrids["full"] - changed_forward9).abs().max()),
                ))
                stacked_outputs = torch.cat((
                    hybrids["native"], hybrids["direct"], hybrids["attention"], hybrids["full"],
                ), dim=0)
                logits = suffix_after_layer(
                    model, next_layer, repeat_batch(x0, 4), repeat_batch(bus9, 4),
                    stacked_outputs, position,
                )
                route_outputs[route] = hybrids
                route_logits[route] = {
                    "native": logits[0], "direct": logits[1],
                    "attention": logits[2], "full": logits[3],
                    "additive": logits[0] + additive_effect(logits[0], logits[1], logits[2]),
                }

            native_logits = route_logits["oracle"]["native"]
            route_rows = {}
            for route in routes:
                logits = route_logits[route]
                outputs = route_outputs[route]
                full_logit_effect = logits["full"] - native_logits
                full_layer9_effect = outputs["full"][0, position] - outputs["native"][0, position]
                direct_layer9_effect = outputs["direct"][0, position] - outputs["native"][0, position]
                attention_layer9_effect = outputs["attention"][0, position] - outputs["native"][0, position]
                additive_layer9_effect = direct_layer9_effect + attention_layer9_effect
                interaction = full_layer9_effect - additive_layer9_effect
                path_rows = {}
                for path in PATHS:
                    effect = logits[path] - native_logits
                    fidelity, cosine, energy = fisher_fidelity(
                        native_logits, full_logit_effect, effect,
                    )
                    if path == "direct":
                        layer9_prediction = direct_layer9_effect
                    elif path == "attention":
                        layer9_prediction = attention_layer9_effect
                    elif path == "additive":
                        layer9_prediction = additive_layer9_effect
                    else:
                        layer9_prediction = full_layer9_effect
                    local_fidelity, local_cosine, _ = euclidean_fidelity(
                        full_layer9_effect, layer9_prediction,
                    )
                    path_rows[path] = {
                        "downstream_fisher_fidelity_to_full": fidelity,
                        "downstream_fisher_cosine_to_full": cosine,
                        "downstream_effect_energy": energy,
                        "layer9_fidelity_to_full": local_fidelity,
                        "layer9_cosine_to_full": local_cosine,
                        "native_kl": exact_kl(native_logits, logits[path]),
                    }
                route_rows[route] = {
                    "paths": path_rows,
                    "layer9_interaction_energy_fraction": float(
                        interaction.square().sum()
                        / full_layer9_effect.square().sum().clamp_min(1e-30)
                    ),
                }

            comparisons = {}
            oracle_logits = route_logits["oracle"]
            for route in routes:
                if route == "oracle":
                    continue
                comparisons[route] = {}
                for path in PATHS:
                    target = oracle_logits[path] - native_logits
                    predicted = route_logits[route][path] - native_logits
                    fidelity, cosine, energy = fisher_fidelity(
                        native_logits, target, predicted,
                    )
                    comparisons[route][path] = {
                        "fisher_fidelity": fidelity,
                        "fisher_cosine": cosine,
                        "oracle_effect_energy": energy,
                    }
            records.append({
                "query": query,
                "row": int(row_index),
                "routes": route_rows,
                "route_to_oracle": comparisons,
            })
        print("query", query, "complete", flush=True)
    max_error = max(recomposition_errors)
    if max_error > 1e-5:
        raise AssertionError(f"layer-9 hybrid recomposition error {max_error:.3e}")
    return {
        "max_native_full_recomposition_error": max_error,
        "summary": summarize(records, routes),
        "records": records,
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    discovery_config = {**config, "positions": config["discovery_positions"]}
    discovery_x, discovery_write = collect(
        model, config["layer"], config["discovery"], discovery_config,
    )
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, :config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :config["queries"]]
    block = model.blocks[config["layer"]]
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("path decomposition", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, values, vectors,
        )

    hypotheses = {
        "P1_additive_paths_explain_full": {
            "supported": all(
                results[c]["summary"]["within_route"][r]["additive"]["median_downstream_fisher_fidelity_to_full"] >= 0.80
                for c in results for r in ("oracle", "hierarchical128")
            ),
        },
        "P2_no_single_path_is_sufficient": {
            "supported": all(
                max(
                    results[c]["summary"]["within_route"]["oracle"][p]["median_downstream_fisher_fidelity_to_full"]
                    for p in ("direct", "attention")
                ) < 0.80 for c in results
            ),
        },
        "P3_immediate_interaction_is_small": {
            "supported": all(
                results[c]["summary"]["within_route"][r]["median_layer9_interaction_energy_fraction"] <= 0.25
                for c in results for r in ("oracle", "hierarchical128")
            ),
        },
        "P4_hierarchical_error_is_attention_heavy": {
            "supported": all(
                results[c]["summary"]["route_to_oracle"]["hierarchical128"]["direct"]["median_fisher_fidelity"]
                - results[c]["summary"]["route_to_oracle"]["hierarchical128"]["attention"]["median_fisher_fidelity"] >= 0.10
                for c in results
            ),
        },
        "P5_flat_improves_both_paths": {
            "supported": all(
                results[c]["summary"]["route_to_oracle"]["flat256"][p]["median_fisher_fidelity"]
                >= results[c]["summary"]["route_to_oracle"]["hierarchical128"][p]["median_fisher_fidelity"]
                for c in results for p in ("direct", "attention")
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Layer-8 → Layer-9 direct and attention path decomposition",
        "",
        "A routed layer-8 MLP input edit changes the layer-8 write. At layer 9, the `direct` hybrid changes the residual/mixed input while clamping the native attention write; the `attention` hybrid changes the attention write while clamping the native mixed input. Layer-9 MLP computation is recomputed in both hybrids, and each hybrid is propagated through layers 10–17 and the unembedding. `Additive` is the sum of the two final-logit effects and is not itself a finite forward pass.",
        "",
        "The first-layer value bus is unchanged by a layer-8 MLP edit in this architecture, so it is held exactly fixed rather than assigned to either path.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Path decomposition](layer8_path_decomposition.png)",
        "",
        "## Within-route decomposition",
        "",
        "| Corpus | Route | Direct fidelity (KL/full) | Attention fidelity (KL/full) | Additive fidelity | Layer-9 interaction energy | Full native KL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        summary = result["summary"]["within_route"]
        for route in config["routes"]:
            row = summary[route]
            lines.append(
                f"| {corpus} | {route} | {row['direct']['median_downstream_fisher_fidelity_to_full']:.3f} "
                f"({row['direct']['median_native_kl_ratio_to_full']:.3f}) | "
                f"{row['attention']['median_downstream_fisher_fidelity_to_full']:.3f} "
                f"({row['attention']['median_native_kl_ratio_to_full']:.3f}) | "
                f"{row['additive']['median_downstream_fisher_fidelity_to_full']:.3f} | "
                f"{row['median_layer9_interaction_energy_fraction']:.3e} | {row['median_full_native_kl']:.3e} |"
            )
    lines += [
        "",
        "## Each routed path compared with the oracle path",
        "",
        "| Corpus | Route | Direct | Attention | Additive | Full |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        comparison = result["summary"]["route_to_oracle"]
        for route in ("flat256", "hierarchical128"):
            lines.append(
                f"| {corpus} | {route} | {comparison[route]['direct']['median_fisher_fidelity']:.3f} | "
                f"{comparison[route]['attention']['median_fisher_fidelity']:.3f} | "
                f"{comparison[route]['additive']['median_fisher_fidelity']:.3f} | "
                f"{comparison[route]['full']['median_fisher_fidelity']:.3f} |"
            )
    lines += [
        "",
        "These are hybrid path-specific causal effects, not percentages of information stored in attention or the MLP. Later nonlinearities mean direct and attention effects need not add, which is precisely what the additive fidelity tests.",
        "",
        "The parenthesized KL/full values compare each hybrid's exact KL from the native distribution with the full edit's KL. KL ratios are magnitude diagnostics, not an additive allocation: direct and attention changes can cooperate or oppose one another.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(3)
    width = 0.36
    paths = ("direct", "attention", "additive")
    for corpus_index, (corpus, result) in enumerate(results.items()):
        offset = (corpus_index - 0.5) * width
        oracle = result["summary"]["within_route"]["oracle"]
        hierarchy = result["summary"]["route_to_oracle"]["hierarchical128"]
        axes[0].bar(x + offset, [oracle[p]["median_downstream_fisher_fidelity_to_full"] for p in paths], width, label=corpus)
        axes[1].bar(x + offset, [hierarchy[p]["median_fisher_fidelity"] for p in paths], width, label=corpus)
    axes[0].set_title("Oracle edit: path sufficiency")
    axes[1].set_title("Hierarchical path vs oracle path")
    for axis in axes:
        axis.set_xticks(x, paths)
        axis.set_ylabel("median Fisher fidelity")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
