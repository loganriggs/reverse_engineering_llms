#!/usr/bin/env python3
"""Exact downstream QK1/QK2/value factorization of semantic attention routes."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers, three_factor_shapley
from run_bilin18_layer8_semantic_attention_causal import edge_mask, median, output_with_attention, pearson
from run_bilin18_layer8_semantic_attention_rank_redteam import all_edge_projections
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_semantic_attention_factor_causality.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_factor_causality.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_FACTOR_CAUSALITY.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_factor_causality.png"
FACTORS = ("qk1", "qk2", "value")
ROUTES = ("frozen64", "oracle64")


def vector_cosine(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator > 1e-30 else 0.0


def absolute_shares(values):
    values = np.abs(np.asarray(values, dtype=np.float64))
    return values / max(values.sum(), 1e-30)


@torch.inference_mode()
def factor_edge_write(block, base, counterfactual, mask, factor_mask):
    """Selected-edge write using counterfactual factors indicated by factor_mask."""
    base_details, counterfactual_details = base["details"], counterfactual["details"]
    score1 = (
        counterfactual_details["score1"].float()
        if factor_mask & 1 else base_details["score1"].float()
    )
    score2 = (
        counterfactual_details["score2"].float()
        if factor_mask & 2 else base_details["score2"].float()
    )
    value = (
        counterfactual_details["mixed_value"].float()
        if factor_mask & 4 else base_details["mixed_value"].float()
    )
    attended = torch.einsum("bhqk,bkhd->bhqd", score1 * score2 * mask, value)
    batch, heads, length, head_dimension = attended.shape
    flattened = attended.transpose(1, 2).contiguous().view(
        batch, length, heads * head_dimension,
    )
    return F.linear(flattened, block.attn.c_proj.weight.float(), bias=None)


@torch.inference_mode()
def local_factor_projections(block, base, counterfactual, mask):
    base_details, counterfactual_details = base["details"], counterfactual["details"]
    a0, a1 = base_details["score1"][0].float(), counterfactual_details["score1"][0].float()
    b0, b1 = base_details["score2"][0].float(), counterfactual_details["score2"][0].float()
    v0 = base_details["mixed_value"][0].float().permute(1, 0, 2)
    v1 = counterfactual_details["mixed_value"][0].float().permute(1, 0, 2)
    components = three_factor_shapley(a0, a1, b0, b1, v0, v1)
    target = counterfactual["attention_write"][0].float() - base["attention_write"][0].float()
    denominator = target.square().sum().clamp_min(1e-30)
    selected_mask = mask[0, ..., None]
    projections = {}
    for name, component in components.items():
        attended = (component * selected_mask).sum(2).transpose(0, 1).contiguous()
        flattened = attended.view(attended.shape[0], -1)
        write = F.linear(flattened, block.attn.c_proj.weight.float(), bias=None)
        projections[name] = float((write * target).sum() / denominator)
    return projections


def factor_shapley(world_logits, direction):
    """Vector Shapley effects for insertion worlds or removal worlds."""
    result = {}
    for factor_index, factor in enumerate(FACTORS):
        bit = 1 << factor_index
        value = torch.zeros_like(world_logits[0])
        for mask in range(8):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            if direction == "insertion":
                marginal = world_logits[mask | bit] - world_logits[mask]
            elif direction == "removal":
                # q(mask) = native_counterfactual - removal_world(mask).
                marginal = world_logits[mask] - world_logits[mask | bit]
            else:
                raise ValueError(direction)
            value = value + weight * marginal
        result[factor] = value
    return result


@torch.inference_mode()
def analyze_route(model, layer, base, counterfactual, edges, final_position):
    block = model.blocks[layer]
    mask = edge_mask(
        model.config.n_head, base["output"].shape[1], edges, base["output"].device,
    )
    edge_writes = [
        factor_edge_write(block, base, counterfactual, mask, factor_mask)
        for factor_mask in range(8)
    ]
    insertion_outputs, removal_outputs = [], []
    for factor_mask in range(8):
        insertion_outputs.append(output_with_attention(
            block,
            base,
            base["attention_write"] + edge_writes[factor_mask] - edge_writes[0],
        ))
        removal_outputs.append(output_with_attention(
            block,
            counterfactual,
            counterfactual["attention_write"]
            + edge_writes[7 ^ factor_mask] - edge_writes[7],
        ))
    outputs = insertion_outputs + removal_outputs
    x0s = [base["x0"]] * 8 + [counterfactual["x0"]] * 8
    buses = [base["value_bus"]] * 8 + [counterfactual["value_bus"]] * 8
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    insertion_logits = logits[:8]
    removal_logits = logits[8:]
    base_logits = insertion_logits[0]
    counterfactual_logits = removal_logits[0]
    target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, target, target)
    insertion_total = insertion_logits[7] - insertion_logits[0]
    removal_total = removal_logits[0] - removal_logits[7]
    insertion_factors = factor_shapley(insertion_logits, "insertion")
    removal_factors = factor_shapley(removal_logits, "removal")

    def projection(effect):
        return fisher_inner(base_logits, effect, target) / max(target_energy, 1e-30)

    def relative_error(factors, total):
        reconstructed = sum(factors.values())
        return float((reconstructed - total).norm() / total.norm().clamp_min(1e-30))

    insertion_projection = {name: projection(value) for name, value in insertion_factors.items()}
    removal_projection = {name: projection(value) for name, value in removal_factors.items()}
    return {
        "edge_count": len(edges),
        "local_factor_projection": local_factor_projections(block, base, counterfactual, mask),
        "insertion": {
            "route_projection": projection(insertion_total),
            "factor_projection": insertion_projection,
            "absolute_factor_share": dict(zip(FACTORS, absolute_shares([
                insertion_projection[name] for name in FACTORS
            ]).tolist())),
            "recomposition_error": relative_error(insertion_factors, insertion_total),
        },
        "removal": {
            "route_projection": projection(removal_total),
            "factor_projection": removal_projection,
            "absolute_factor_share": dict(zip(FACTORS, absolute_shares([
                removal_projection[name] for name in FACTORS
            ]).tolist())),
            "recomposition_error": relative_error(removal_factors, removal_total),
        },
    }


def summarize(records, layers):
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        selected = [record["layers"][layer_index] for record in records]
        route_rows = {}
        for route in ROUTES:
            rows = [row[route] for row in selected]
            route_rows[route] = {
                "local_factor_projection": {
                    factor: median([row["local_factor_projection"][factor] for row in rows])
                    for factor in FACTORS
                },
            }
            for direction in ("insertion", "removal"):
                route_rows[route][direction] = {
                    "route_projection": median([row[direction]["route_projection"] for row in rows]),
                    "factor_projection": {
                        factor: median([row[direction]["factor_projection"][factor] for row in rows])
                        for factor in FACTORS
                    },
                    "absolute_factor_share": {
                        factor: median([row[direction]["absolute_factor_share"][factor] for row in rows])
                        for factor in FACTORS
                    },
                    "maximum_recomposition_error": max(
                        row[direction]["recomposition_error"] for row in rows
                    ),
                }
        layer_rows.append({"layer": layer, "routes": route_rows})

    local, insertion, removal = [], [], []
    for record in records:
        for layer_row in record["layers"]:
            route = layer_row["frozen64"]
            for factor in FACTORS:
                local.append(route["local_factor_projection"][factor])
                insertion.append(route["insertion"]["factor_projection"][factor])
                removal.append(route["removal"]["factor_projection"][factor])
    return {
        "layers": layer_rows,
        "correlations": {
            "local_to_downstream_removal": pearson(local, removal),
            "local_to_downstream_insertion": pearson(local, insertion),
            "downstream_insertion_to_removal": pearson(insertion, removal),
        },
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    frozen_config = json.loads(Path(config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    frozen = json.loads(Path(config["frozen_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    frozen_edges = {
        layer: [
            tuple(row["edge"])
            for row in frozen["discovered_routes"][str(layer)]["magnitude"][:config["route_rank"]]
        ]
        for layer in config["layers"]
    }
    results, records_by_corpus = {}, {}
    maximum_error = 0.0

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                row_index, start = windows[config["evaluation_context_index"]]
                base_raw = np.asarray(
                    rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                ).copy()
                counterfactual_raw = base_raw.copy()
                counterfactual_raw[edit_position] = target_token
                base_layers = forward_layers(
                    model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=max(config["layers"]),
                )
                counterfactual_layers = forward_layers(
                    model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=max(config["layers"]),
                )
                layer_rows = []
                for layer in config["layers"]:
                    ranked = all_edge_projections(
                        model.blocks[layer], base_layers[layer], counterfactual_layers[layer],
                        edit_position, final_position,
                    )
                    oracle_edges = [(row[1], row[2], row[3]) for row in ranked[:config["route_rank"]]]
                    row = {
                        "frozen64": analyze_route(
                            model, layer, base_layers[layer], counterfactual_layers[layer],
                            frozen_edges[layer], final_position,
                        ),
                        "oracle64": analyze_route(
                            model, layer, base_layers[layer], counterfactual_layers[layer],
                            oracle_edges, final_position,
                        ),
                    }
                    maximum_error = max(
                        maximum_error,
                        *(row[route][direction]["recomposition_error"]
                          for route, direction in itertools.product(ROUTES, ("insertion", "removal"))),
                    )
                    layer_rows.append(row)
                records.append({
                    "family": family,
                    "pair": f"{source_token}->{target_token}",
                    "row": int(row_index),
                    "source_start": int(start),
                    "layers": layer_rows,
                })
                print(corpus, family, source_token, "->", target_token, flush=True)
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records, config["layers"])

    layer4_index = config["layers"].index(4)
    layer4_shares = {
        corpus: np.asarray([
            results[corpus]["layers"][layer4_index]["routes"]["frozen64"]["removal"]
            ["absolute_factor_share"][factor]
            for factor in FACTORS
        ])
        for corpus in results
    }
    factor_share_l1 = float(np.abs(layer4_shares["fineweb"] - layer4_shares["tinystories"]).sum())
    frozen_oracle_cosine = {}
    for corpus in results:
        layer_routes = results[corpus]["layers"][layer4_index]["routes"]
        frozen_vector = [
            layer_routes["frozen64"]["removal"]["factor_projection"][factor]
            for factor in FACTORS
        ]
        oracle_vector = [
            layer_routes["oracle64"]["removal"]["factor_projection"][factor]
            for factor in FACTORS
        ]
        frozen_oracle_cosine[corpus] = vector_cosine(frozen_vector, oracle_vector)

    hypotheses = {
        "T1_factor_shapley_recomposes_routes": {
            "supported": maximum_error <= 1e-5,
            "maximum_relative_error": maximum_error,
        },
        "T2_value_content_dominates_layer4": {
            "supported": all(layer4_shares[corpus][2] >= 0.50 for corpus in results),
        },
        "T3_qk_routing_is_nontrivial": {
            "supported": all(layer4_shares[corpus][:2].sum() >= 0.15 for corpus in results),
        },
        "T4_local_factor_mix_predicts_downstream_mix": {
            "supported": all(
                results[corpus]["correlations"]["local_to_downstream_removal"] >= 0.50
                for corpus in results
            ),
        },
        "T5_factor_mix_transfers_ood": {
            "supported": factor_share_l1 <= 0.25,
            "layer4_absolute_share_l1": factor_share_l1,
        },
        "T6_factor_attribution_is_cross_world_stable": {
            "supported": all(
                results[corpus]["correlations"]["downstream_insertion_to_removal"] >= 0.70
                for corpus in results
            ),
        },
        "T7_frozen_and_oracle_use_similar_factors": {
            "supported": all(value >= 0.80 for value in frozen_oracle_cosine.values()),
            "cosines": frozen_oracle_cosine,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_recomposition_error": maximum_error,
        "layer4_absolute_factor_shares": {
            corpus: dict(zip(FACTORS, shares.tolist()))
            for corpus, shares in layer4_shares.items()
        },
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Causal QK1/QK2/value factorization of semantic attention routes",
        "",
        "For the frozen-64 and per-example oracle-64 edge routes, all eight QK1/QK2/value hybrid worlds are constructed exactly. Each world replaces only the selected edge terms, recomputes the target bilinear MLP, and runs the complete suffix. Vector-valued Shapley effects are computed separately for insertion into the base context and removal from the counterfactual context.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Attention factor causality](layer8_semantic_attention_factor_causality.png)",
        "",
        "| Corpus | Layer | Route | Removal route projection | QK1 / QK2 / value Shapley projection | Absolute shares |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for layer_row in result["layers"]:
            for route in ROUTES:
                row = layer_row["routes"][route]["removal"]
                projection = row["factor_projection"]
                shares = row["absolute_factor_share"]
                lines.append(
                    f"| {corpus} | {layer_row['layer']} | {route} | {row['route_projection']:.3f} | "
                    f"{projection['qk1']:.3f} / {projection['qk2']:.3f} / {projection['value']:.3f} | "
                    f"{shares['qk1']:.3f} / {shares['qk2']:.3f} / {shares['value']:.3f} |"
                )
    lines += [
        "",
        "| Corpus | Local→removal factor correlation | Local→insertion | Insertion→removal | Frozen/oracle cosine at L4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        correlations = result["correlations"]
        lines.append(
            f"| {corpus} | {correlations['local_to_downstream_removal']:.3f} | "
            f"{correlations['local_to_downstream_insertion']:.3f} | "
            f"{correlations['downstream_insertion_to_removal']:.3f} | "
            f"{frozen_oracle_cosine[corpus]:.3f} |"
        )
    lines += [
        "",
        f"Maximum vector Shapley recomposition error: `{maximum_error:.2e}`. Layer-4 absolute-share L1 distance across corpora: `{factor_share_l1:.3f}`.",
        "",
        "These Shapley values allocate the effect of a declared cross-world edge replacement. They are exact for that intervention game, but they are not unique percentages of where information is stored: changing the background world or the coalition definition changes the causal question.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    width = 0.23
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        x = np.arange(len(config["layers"]))
        for factor_index, factor in enumerate(FACTORS):
            axis.bar(
                x + (factor_index - 1) * width,
                [
                    results[corpus]["layers"][index]["routes"]["frozen64"]["removal"]
                    ["factor_projection"][factor]
                    for index in range(len(config["layers"]))
                ],
                width,
                label=factor,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xticks(x, config["layers"])
        axis.set_title(corpus)
        axis.set_xlabel("layer")
        axis.set_ylabel("downstream target projection")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
