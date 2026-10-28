#!/usr/bin/env python3
"""Split Layer-4 current value into incoming residual and repeated embedding paths."""

from __future__ import annotations

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


CONFIG = Path("configs/bilin18_layer4_embedding_residual_causality.json")
OUTPUT = TENSOR_ROOT / "layer4_embedding_residual_causality.json"
REPORT = TENSOR_ROOT / "LAYER4_EMBEDDING_RESIDUAL_CAUSALITY.md"
FIGURE = TENSOR_ROOT / "layer4_embedding_residual_causality.png"
FACTORS = ("qk1", "qk2", "residual_state", "embedding_stream", "value_bus")
ROUTES = ("frozen64", "oracle64")


def recover_residual_input(block, record):
    return (
        record["mixed"] - block.lambdas[1].float() * record["x0"]
    ) / block.lambdas[0].float()


@torch.inference_mode()
def five_factor_edge_write(block, base, counterfactual, mask, factor_mask):
    base_details, counterfactual_details = base["details"], counterfactual["details"]
    score1 = (
        counterfactual_details["score1"].float()
        if factor_mask & 1 else base_details["score1"].float()
    )
    score2 = (
        counterfactual_details["score2"].float()
        if factor_mask & 2 else base_details["score2"].float()
    )
    base_residual = recover_residual_input(block, base)
    counterfactual_residual = recover_residual_input(block, counterfactual)
    residual = counterfactual_residual if factor_mask & 4 else base_residual
    embedding = counterfactual["x0"] if factor_mask & 8 else base["x0"]
    mixed = block.lambdas[0].float() * residual + block.lambdas[1].float() * embedding
    normalized = F.rms_norm(mixed, (mixed.shape[-1],))
    batch, length, dimension = normalized.shape
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    current_value = block.attn.c_v(normalized).view(batch, length, heads, head_dimension)
    value_bus = (
        counterfactual_details["value_bus"].float()
        if factor_mask & 16 else base_details["value_bus"].float()
    )
    mixed_value = (
        (1 - block.attn.lamb.float()) * current_value
        + block.attn.lamb.float() * value_bus
    )
    attended = torch.einsum("bhqk,bkhd->bhqd", score1 * score2 * mask, mixed_value)
    flattened = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
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
        five_factor_edge_write(block, base, counterfactual, mask, factor_mask)
        for factor_mask in range(32)
    ]
    insertion_outputs, removal_outputs = [], []
    for factor_mask in range(32):
        insertion_outputs.append(output_with_attention(
            block, base,
            base["attention_write"] + edge_writes[factor_mask] - edge_writes[0],
        ))
        removal_outputs.append(output_with_attention(
            block, counterfactual,
            counterfactual["attention_write"]
            + edge_writes[31 ^ factor_mask] - edge_writes[31],
        ))
    outputs = insertion_outputs + removal_outputs
    x0s = [base["x0"]] * 32 + [counterfactual["x0"]] * 32
    buses = [base["value_bus"]] * 32 + [counterfactual["value_bus"]] * 32
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    insertion_logits, removal_logits = logits[:32], logits[32:]
    base_logits, counterfactual_logits = insertion_logits[0], removal_logits[0]
    target = counterfactual_logits - base_logits
    energy = fisher_inner(base_logits, target, target)
    insertion_total = insertion_logits[31] - insertion_logits[0]
    removal_total = removal_logits[0] - removal_logits[31]
    insertion_factors = factor_shapley(insertion_logits, "insertion")
    removal_factors = factor_shapley(removal_logits, "removal")

    def projection(effect):
        return fisher_inner(base_logits, effect, target) / max(energy, 1e-30)

    def summarize_direction(factors, total):
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
            "construction_cancellation_ratio": abs(
                projections["residual_state"] + projections["embedding_stream"]
            ) / max(
                abs(projections["residual_state"]) + abs(projections["embedding_stream"]),
                1e-30,
            ),
        }

    return {
        "edge_count": len(edges),
        "block_residual_coefficient": float(block.lambdas[0]),
        "block_embedding_coefficient": float(block.lambdas[1]),
        "attention_value_bus_lambda": float(block.attn.lamb),
        "insertion": summarize_direction(insertion_factors, insertion_total),
        "removal": summarize_direction(removal_factors, removal_total),
    }


def summarize(records):
    routes = {}
    for route in ROUTES:
        rows = [record[route] for record in records]
        routes[route] = {}
        for direction in ("insertion", "removal"):
            routes[route][direction] = {
                "route_projection": median([row[direction]["route_projection"] for row in rows]),
                "factor_projection": {
                    factor: median([row[direction]["factor_projection"][factor] for row in rows])
                    for factor in FACTORS
                },
                "absolute_factor_share": {
                    factor: median([row[direction]["absolute_factor_share"][factor] for row in rows])
                    for factor in FACTORS
                },
                "construction_cancellation_ratio": median([
                    row[direction]["construction_cancellation_ratio"] for row in rows
                ]),
                "maximum_recomposition_error": max(
                    row[direction]["recomposition_error"] for row in rows
                ),
            }
    insertion = [
        record["frozen64"]["insertion"]["factor_projection"][factor]
        for record in records for factor in FACTORS
    ]
    removal = [
        record["frozen64"]["removal"]["factor_projection"][factor]
        for record in records for factor in FACTORS
    ]
    first = records[0]["frozen64"]
    return {
        "coefficients": {
            "residual": first["block_residual_coefficient"],
            "embedding": first["block_embedding_coefficient"],
            "value_bus": first["attention_value_bus_lambda"],
        },
        "routes": routes,
        "insertion_to_removal_correlation": pearson(insertion, removal),
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    recurrence_config = json.loads(Path(config["source_config"]).read_text())
    value_config = json.loads(Path(recurrence_config["source_config"]).read_text())
    factor_config = json.loads(Path(value_config["source_config"]).read_text())
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
    layer = config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    frozen_edges = [
        tuple(row["edge"])
        for row in frozen["discovered_routes"][str(layer)]["magnitude"][:config["route_rank"]]
    ]
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
                base = forward_layers(
                    model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer,
                )[layer]
                counterfactual = forward_layers(
                    model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=layer,
                )[layer]
                ranked = all_edge_projections(
                    model.blocks[layer], base, counterfactual, edit_position, final_position,
                )
                oracle_edges = [(row[1], row[2], row[3]) for row in ranked[:config["route_rank"]]]
                record = {
                    "family": family,
                    "pair": f"{source_token}->{target_token}",
                    "row": int(row_index),
                    "source_start": int(start),
                    "frozen64": analyze_route(
                        model, layer, base, counterfactual, frozen_edges, final_position,
                    ),
                    "oracle64": analyze_route(
                        model, layer, base, counterfactual, oracle_edges, final_position,
                    ),
                }
                maximum_error = max(
                    maximum_error,
                    *(record[route][direction]["recomposition_error"]
                      for route in ROUTES for direction in ("insertion", "removal")),
                )
                records.append(record)
                print(corpus, family, source_token, "->", target_token, flush=True)
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records)

    shares = {
        corpus: np.asarray([
            results[corpus]["routes"]["frozen64"]["removal"]["absolute_factor_share"][factor]
            for factor in FACTORS
        ])
        for corpus in results
    }
    share_l1 = float(np.abs(shares["fineweb"] - shares["tinystories"]).sum())
    frozen_oracle_cosines = {}
    for corpus in results:
        routes = results[corpus]["routes"]
        frozen_vector = [
            routes["frozen64"]["removal"]["factor_projection"][factor] for factor in FACTORS
        ]
        oracle_vector = [
            routes["oracle64"]["removal"]["factor_projection"][factor] for factor in FACTORS
        ]
        frozen_oracle_cosines[corpus] = vector_cosine(frozen_vector, oracle_vector)

    hypotheses = {
        "W1_five_factor_shapley_recomposes_routes": {
            "supported": maximum_error <= 1e-5,
            "maximum_relative_error": maximum_error,
        },
        "W2_embedding_injection_dominates": {
            "supported": all(shares[corpus][3] >= 0.50 for corpus in results),
        },
        "W3_residual_state_is_nontrivial": {
            "supported": all(shares[corpus][2] >= 0.15 for corpus in results),
        },
        "W4_current_value_construction_dominates": {
            "supported": all(shares[corpus][2:4].sum() >= 0.60 for corpus in results),
        },
        "W5_embedding_and_residual_do_not_cancel": {
            "supported": all(
                results[corpus]["routes"]["frozen64"]["removal"]["construction_cancellation_ratio"] >= 0.50
                for corpus in results
            ),
        },
        "W6_construction_mix_transfers_ood": {
            "supported": share_l1 <= 0.25,
            "absolute_share_l1": share_l1,
        },
        "W7_frozen_and_oracle_construction_match": {
            "supported": all(value >= 0.80 for value in frozen_oracle_cosines.values()),
            "cosines": frozen_oracle_cosines,
        },
        "W8_construction_attribution_is_cross_world_stable": {
            "supported": all(
                results[corpus]["insertion_to_removal_correlation"] >= 0.70
                for corpus in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_recomposition_error": maximum_error,
        "absolute_factor_shares": {
            corpus: dict(zip(FACTORS, values.tolist())) for corpus, values in shares.items()
        },
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Repeated embedding versus residual construction of Layer-4 value content",
        "",
        "The dominant current-value path is split into the incoming residual state and the stable token embedding injected again at Layer 4. Along with QK1, QK2, and the inherited layer-0 value bus, this defines 32 exact hybrid worlds. RMSNorm and the current value projection are recomputed for every residual/embedding coalition before the layer MLP and full suffix are run.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Embedding/residual causality](layer4_embedding_residual_causality.png)",
        "",
        "| Corpus | Route | Removal route | QK1 / QK2 / residual / embedding / bus projection | Absolute shares | Residual+embedding cancellation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for route in ROUTES:
            row = result["routes"][route]["removal"]
            projection, factor_shares = row["factor_projection"], row["absolute_factor_share"]
            lines.append(
                f"| {corpus} | {route} | {row['route_projection']:.3f} | "
                + " / ".join(f"{projection[factor]:.3f}" for factor in FACTORS)
                + " | "
                + " / ".join(f"{factor_shares[factor]:.3f}" for factor in FACTORS)
                + f" | {row['construction_cancellation_ratio']:.3f} |"
            )
    lines += [
        "",
        "| Corpus | Residual / embedding / value-bus coefficients | Insertion→removal | Frozen/oracle cosine |",
        "|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        coefficients = result["coefficients"]
        lines.append(
            f"| {corpus} | {coefficients['residual']:.3f} / {coefficients['embedding']:.3f} / {coefficients['value_bus']:.3f} | "
            f"{result['insertion_to_removal_correlation']:.3f} | {frozen_oracle_cosines[corpus]:.3f} |"
        )
    lines += [
        "",
        f"Maximum vector Shapley recomposition error: `{maximum_error:.2e}`. Cross-corpus absolute-share L1 distance: `{share_l1:.3f}`.",
        "",
        "This separates causal input paths under the declared coalition game. It does not by itself say whether the embedding geometry was learned by this model, inherited from tied training dynamics, or semantically sufficient without contextual residual processing.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    width = 0.16
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        route = results[corpus]["routes"]["frozen64"]["removal"]
        values = [route["factor_projection"][factor] for factor in FACTORS]
        axis.bar(np.arange(len(FACTORS)), values)
        axis.set_xticks(np.arange(len(FACTORS)), ["QK1", "QK2", "residual", "embedding", "bus"], rotation=20)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(corpus)
        axis.set_ylabel("downstream target projection")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
