#!/usr/bin/env python3
"""Split semantic attention routes into QK1, QK2, current-V, and value-bus paths."""

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
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import edge_mask, median, output_with_attention, pearson
from run_bilin18_layer8_semantic_attention_factor_causality import absolute_shares, vector_cosine
from run_bilin18_layer8_semantic_attention_rank_redteam import all_edge_projections
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_semantic_attention_value_bus_causality.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_value_bus_causality.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_VALUE_BUS_CAUSALITY.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_value_bus_causality.png"
FACTORS = ("qk1", "qk2", "current_value", "value_bus")
ROUTES = ("frozen64", "oracle64")


@torch.inference_mode()
def four_factor_edge_write(block, base, counterfactual, mask, factor_mask):
    base_details, counterfactual_details = base["details"], counterfactual["details"]

    def choose(name, bit):
        return (
            counterfactual_details[name].float()
            if factor_mask & bit else base_details[name].float()
        )

    score1 = choose("score1", 1)
    score2 = choose("score2", 2)
    current_value = choose("current_value", 4)
    value_bus = choose("value_bus", 8)
    mixed_value = (1 - block.attn.lamb.float()) * current_value + block.attn.lamb.float() * value_bus
    attended = torch.einsum("bhqk,bkhd->bhqd", score1 * score2 * mask, mixed_value)
    batch, heads, length, head_dimension = attended.shape
    flattened = attended.transpose(1, 2).contiguous().view(
        batch, length, heads * head_dimension,
    )
    return F.linear(flattened, block.attn.c_proj.weight.float(), bias=None)


def factor_shapley(world_values, direction):
    result = {}
    count = len(FACTORS)
    for factor_index, factor in enumerate(FACTORS):
        bit = 1 << factor_index
        value = torch.zeros_like(world_values[0])
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = (
                math.factorial(size) * math.factorial(count - size - 1)
                / math.factorial(count)
            )
            if direction == "insertion":
                marginal = world_values[mask | bit] - world_values[mask]
            elif direction == "removal":
                marginal = world_values[mask] - world_values[mask | bit]
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
        four_factor_edge_write(block, base, counterfactual, mask, factor_mask)
        for factor_mask in range(16)
    ]
    local_target = counterfactual["attention_write"].float() - base["attention_write"].float()
    local_energy = float(local_target.square().sum().clamp_min(1e-30))
    local_effects = factor_shapley(edge_writes, "insertion")
    local_projection = {
        name: float((effect * local_target).sum() / local_energy)
        for name, effect in local_effects.items()
    }

    insertion_outputs, removal_outputs = [], []
    for factor_mask in range(16):
        insertion_outputs.append(output_with_attention(
            block,
            base,
            base["attention_write"] + edge_writes[factor_mask] - edge_writes[0],
        ))
        removal_outputs.append(output_with_attention(
            block,
            counterfactual,
            counterfactual["attention_write"]
            + edge_writes[15 ^ factor_mask] - edge_writes[15],
        ))
    outputs = insertion_outputs + removal_outputs
    x0s = [base["x0"]] * 16 + [counterfactual["x0"]] * 16
    buses = [base["value_bus"]] * 16 + [counterfactual["value_bus"]] * 16
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    insertion_logits, removal_logits = logits[:16], logits[16:]
    base_logits, counterfactual_logits = insertion_logits[0], removal_logits[0]
    target = counterfactual_logits - base_logits
    energy = fisher_inner(base_logits, target, target)
    insertion_total = insertion_logits[15] - insertion_logits[0]
    removal_total = removal_logits[0] - removal_logits[15]
    insertion_factors = factor_shapley(insertion_logits, "insertion")
    removal_factors = factor_shapley(removal_logits, "removal")

    def projection(effect):
        return fisher_inner(base_logits, effect, target) / max(energy, 1e-30)

    def direction_result(factors, total):
        projections = {name: projection(effect) for name, effect in factors.items()}
        shares = absolute_shares([projections[name] for name in FACTORS])
        reconstruction = sum(factors.values())
        return {
            "route_projection": projection(total),
            "factor_projection": projections,
            "absolute_factor_share": dict(zip(FACTORS, shares.tolist())),
            "recomposition_error": float(
                (reconstruction - total).norm() / total.norm().clamp_min(1e-30)
            ),
            "cancellation_ratio": abs(
                projections["current_value"] + projections["value_bus"]
            ) / max(
                abs(projections["current_value"]) + abs(projections["value_bus"]),
                1e-30,
            ),
        }

    return {
        "edge_count": len(edges),
        "attention_value_bus_lambda": float(block.attn.lamb),
        "local_factor_projection": local_projection,
        "insertion": direction_result(insertion_factors, insertion_total),
        "removal": direction_result(removal_factors, removal_total),
    }


def summarize(records, layers):
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        selected = [record["layers"][layer_index] for record in records]
        route_rows = {}
        for route in ROUTES:
            rows = [row[route] for row in selected]
            route_rows[route] = {
                "attention_value_bus_lambda": median([
                    row["attention_value_bus_lambda"] for row in rows
                ]),
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
                    "cancellation_ratio": median([
                        row[direction]["cancellation_ratio"] for row in rows
                    ]),
                    "maximum_recomposition_error": max(
                        row[direction]["recomposition_error"] for row in rows
                    ),
                }
        layer_rows.append({"layer": layer, "routes": route_rows})

    insertion, removal, local = [], [], []
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
            "downstream_insertion_to_removal": pearson(insertion, removal),
        },
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    factor_config = json.loads(Path(config["source_config"]).read_text())
    frozen_config = json.loads(Path(factor_config["source_config"]).read_text())
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
    shares = {
        corpus: np.asarray([
            results[corpus]["layers"][layer4_index]["routes"]["frozen64"]["removal"]
            ["absolute_factor_share"][factor]
            for factor in FACTORS
        ])
        for corpus in results
    }
    share_l1 = float(np.abs(shares["fineweb"] - shares["tinystories"]).sum())
    frozen_oracle_cosines = {}
    for corpus in results:
        routes = results[corpus]["layers"][layer4_index]["routes"]
        frozen_vector = [
            routes["frozen64"]["removal"]["factor_projection"][factor]
            for factor in FACTORS
        ]
        oracle_vector = [
            routes["oracle64"]["removal"]["factor_projection"][factor]
            for factor in FACTORS
        ]
        frozen_oracle_cosines[corpus] = vector_cosine(frozen_vector, oracle_vector)

    hypotheses = {
        "U1_four_factor_shapley_recomposes_routes": {
            "supported": maximum_error <= 1e-5,
            "maximum_relative_error": maximum_error,
        },
        "U2_both_value_paths_are_substantial": {
            "supported": all(shares[corpus][2] >= 0.20 and shares[corpus][3] >= 0.20 for corpus in results),
        },
        "U3_value_paths_cancel": {
            "supported": all(
                results[corpus]["layers"][layer4_index]["routes"]["frozen64"]["removal"]["cancellation_ratio"] <= 0.50
                for corpus in results
            ),
        },
        "U4_value_path_mix_transfers_ood": {
            "supported": share_l1 <= 0.25,
            "layer4_absolute_share_l1": share_l1,
        },
        "U5_value_path_attribution_is_cross_world_stable": {
            "supported": all(
                results[corpus]["correlations"]["downstream_insertion_to_removal"] >= 0.70
                for corpus in results
            ),
        },
        "U6_frozen_and_oracle_value_paths_match": {
            "supported": all(value >= 0.80 for value in frozen_oracle_cosines.values()),
            "cosines": frozen_oracle_cosines,
        },
        "U7_qk_remains_secondary": {
            "supported": all(shares[corpus][:2].sum() <= 0.20 for corpus in results),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_recomposition_error": maximum_error,
        "layer4_absolute_factor_shares": {
            corpus: dict(zip(FACTORS, values.tolist())) for corpus, values in shares.items()
        },
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Current-value versus inherited-bus causality in semantic attention routes",
        "",
        "The earlier value factor is split into the target layer's current value projection and the value bus inherited from layer 0. Together with QK1 and QK2 this gives sixteen exact hybrid worlds. The selected edge write is `(QK1)(QK2)[(1−λ)V_current + λV_bus]`; complete layer MLP and suffix recomputation follows every hybrid.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Value-bus factor causality](layer8_semantic_attention_value_bus_causality.png)",
        "",
        "| Corpus | Layer | Route | λ | Removal route | QK1 / QK2 / current-V / bus-V projection | Absolute shares | V cancellation |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for layer_row in result["layers"]:
            for route in ROUTES:
                route_row = layer_row["routes"][route]
                row = route_row["removal"]
                projection = row["factor_projection"]
                factor_shares = row["absolute_factor_share"]
                lines.append(
                    f"| {corpus} | {layer_row['layer']} | {route} | {route_row['attention_value_bus_lambda']:.3f} | "
                    f"{row['route_projection']:.3f} | "
                    f"{projection['qk1']:.3f} / {projection['qk2']:.3f} / "
                    f"{projection['current_value']:.3f} / {projection['value_bus']:.3f} | "
                    f"{factor_shares['qk1']:.3f} / {factor_shares['qk2']:.3f} / "
                    f"{factor_shares['current_value']:.3f} / {factor_shares['value_bus']:.3f} | "
                    f"{row['cancellation_ratio']:.3f} |"
                )
    lines += [
        "",
        "| Corpus | Local→downstream factor correlation | Insertion→removal | Frozen/oracle cosine at L4 |",
        "|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        lines.append(
            f"| {corpus} | {result['correlations']['local_to_downstream_removal']:.3f} | "
            f"{result['correlations']['downstream_insertion_to_removal']:.3f} | "
            f"{frozen_oracle_cosines[corpus]:.3f} |"
        )
    lines += [
        "",
        f"Maximum vector Shapley recomposition error: `{maximum_error:.2e}`. Layer-4 four-factor share L1 distance across corpora: `{share_l1:.3f}`.",
        "",
        "A low cancellation ratio means the two value paths have opposing target projections under this intervention game. It does not imply either path can be deleted safely: their difference is the implemented operation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    width = 0.18
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        x = np.arange(len(config["layers"]))
        for factor_index, factor in enumerate(FACTORS):
            axis.bar(
                x + (factor_index - 1.5) * width,
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
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
