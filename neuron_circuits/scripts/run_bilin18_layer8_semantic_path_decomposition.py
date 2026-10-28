#!/usr/bin/env python3
"""Factor semantic counterfactuals across Layer-8 tensor-network carriers."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_router_causal_validation import exact_kl, prefix_to_layer, suffix_from_layer
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_path_decomposition.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_path_decomposition.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_PATH_DECOMPOSITION.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_path_decomposition.png"
COMPONENTS = ("residual_sequence", "embedding_stream", "value_bus")


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def fisher_inner(reference_logits, first, second):
    probability = F.softmax(reference_logits.float(), -1)
    first = first.float() - (probability * first.float()).sum()
    second = second.float() - (probability * second.float()).sum()
    return float((probability * first * second).sum())


@torch.inference_mode()
def boundary_state(model, block, ids, position):
    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
    write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    return {
        "x0": x0,
        "value_bus": value_bus,
        "pre": pre,
        "normalized": normalized,
        "write": write,
        "output": pre + write,
    }


@torch.inference_mode()
def analyze_paths(model, block, base_ids, counterfactual_ids, position):
    base = boundary_state(model, block, base_ids, position)
    counterfactual = boundary_state(model, block, counterfactual_ids, position)
    masks = list(range(8))
    outputs, x0s, buses = [], [], []
    for mask in masks:
        outputs.append(counterfactual["output"] if mask & 1 else base["output"])
        x0s.append(counterfactual["x0"] if mask & 2 else base["x0"])
        buses.append(counterfactual["value_bus"] if mask & 4 else base["value_bus"])
    current_output = base["output"].clone()
    current_output[0, position] = counterfactual["output"][0, position]
    outputs.append(current_output)
    x0s.append(base["x0"])
    buses.append(base["value_bus"])
    mlp_output = base["output"].clone()
    mlp_output[0, position] = base["pre"][0, position] + counterfactual["write"][0, position]
    outputs.append(mlp_output)
    x0s.append(base["x0"])
    buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, 8, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), position,
    )
    effects = {mask: logits[mask] - logits[0] for mask in masks}
    target = effects[7]
    target_energy = fisher_inner(logits[0], target, target)
    routes = {}
    route_indices = {
        "residual_only": 1,
        "embedding_only": 2,
        "bus_only": 4,
        "residual_embedding": 3,
        "residual_bus": 5,
        "embedding_bus": 6,
        "full_boundary": 7,
        "current_position_only": 8,
        "final_mlp_write_only": 9,
    }
    for name, index in route_indices.items():
        effect = logits[index] - logits[0]
        fidelity, cosine, _ = fisher_fidelity(logits[0], target, effect)
        routes[name] = {
            "fidelity": fidelity,
            "cosine": cosine,
            "kl": exact_kl(logits[0], logits[index]),
            "kl_to_full": exact_kl(logits[0], logits[index]) / max(exact_kl(logits[0], logits[7]), 1e-30),
            "target_projection": fisher_inner(logits[0], effect, target) / max(target_energy, 1e-30),
        }
    shapley = {}
    n = len(COMPONENTS)
    for component_index, component in enumerate(COMPONENTS):
        value = 0.0
        bit = 1 << component_index
        for mask in masks:
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            marginal = effects[mask | bit] - effects[mask]
            value += weight * fisher_inner(logits[0], marginal, target) / max(target_energy, 1e-30)
        shapley[component] = value
    return {
        "full_kl": routes["full_boundary"]["kl"],
        "routes": routes,
        "shapley_target_projection": shapley,
        "shapley_sum": sum(shapley.values()),
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    parent = json.loads(Path(source["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[8]
    position = source["causal_position"]
    results = {}
    maximum_shapley_error = 0.0

    for corpus_index, (corpus, corpus_config) in enumerate(source["evaluation"].items()):
        raw_source = np.load(corpus_config["path"], mmap_mode="r")
        rows = raw_source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, source["edit_position"], source["causal_position"],
                    config["contexts_per_pair"], rng,
                )
                for row_index, start in windows:
                    base_raw = np.asarray(
                        rows[row_index, start:start + source["sequence_length"]], dtype=np.int64,
                    ).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[source["edit_position"]] = target_token
                    record = analyze_paths(
                        model, block,
                        torch.as_tensor(base_raw, device="cuda")[None],
                        torch.as_tensor(counterfactual_raw, device="cuda")[None], position,
                    )
                    record.update({"family": family, "pair": f"{source_token}->{target_token}", "row": row_index})
                    maximum_shapley_error = max(maximum_shapley_error, abs(record["shapley_sum"] - 1))
                    records.append(record)
                print(corpus, family, source_token, "->", target_token, flush=True)
        route_summary = {
            route: {
                key: median([record["routes"][route][key] for record in records])
                for key in ("fidelity", "cosine", "kl", "kl_to_full", "target_projection")
            }
            for route in records[0]["routes"]
        }
        shapley_summary = {
            component: median([record["shapley_target_projection"][component] for record in records])
            for component in COMPONENTS
        }
        results[corpus] = {
            "routes": route_summary,
            "shapley_target_projection": shapley_summary,
            "records": records,
        }

    hypotheses = {
        "H1_full_boundary_recomposition_is_exact": {"supported": all(results[c]["routes"]["full_boundary"]["fidelity"] >= 0.999 for c in results)},
        "H2_residual_sequence_is_major_carrier": {"supported": all(results[c]["routes"]["residual_only"]["fidelity"] >= 0.50 for c in results)},
        "H3_current_position_is_not_sufficient": {"supported": all(results[c]["routes"]["current_position_only"]["fidelity"] < 0.50 for c in results)},
        "H4_prior_positions_matter": {"supported": all(results[c]["routes"]["residual_only"]["fidelity"] - results[c]["routes"]["current_position_only"]["fidelity"] >= 0.20 for c in results)},
        "H5_nonresidual_carriers_matter": {"supported": all(abs(results[c]["shapley_target_projection"]["embedding_stream"]) + abs(results[c]["shapley_target_projection"]["value_bus"]) >= 0.20 for c in results)},
        "H6_shapley_recomposes_target": {"supported": maximum_shapley_error <= 1e-5},
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "maximum_shapley_sum_error": maximum_shapley_error,
        "results": results,
    }, indent=2))

    lines = [
        "# Layer-8 boundary paths for semantic counterfactuals",
        "",
        "The fixed-position natural counterfactual panel is recomposed through all eight combinations of three Layer-8 boundary carriers: the entire residual sequence, the stable embedding stream injected at later blocks, and the attention value bus. Current-position residual and final-MLP-write-only patches are additional controls. Signed Shapley values allocate projection onto the exact counterfactual logit effect and sum exactly even when paths interact.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic path decomposition](layer8_semantic_path_decomposition.png)",
        "",
        "| Corpus | Route | Fidelity / cosine | KL/full | Target projection |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for route, row in result["routes"].items():
            lines.append(
                f"| {corpus} | {route} | {row['fidelity']:.3f} / {row['cosine']:.3f} | "
                f"{row['kl_to_full']:.3f} | {row['target_projection']:.3f} |"
            )
    lines += [
        "",
        "| Corpus | Residual Shapley | Embedding Shapley | Value-bus Shapley |",
        "|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        row = result["shapley_target_projection"]
        lines.append(f"| {corpus} | {row['residual_sequence']:.3f} | {row['embedding_stream']:.3f} | {row['value_bus']:.3f} |")
    lines += [
        "",
        f"Maximum per-edit Shapley sum error: `{maximum_shapley_error:.2e}`.",
        "",
        "These are boundary-carrier attributions, not percentages of information stored. A carrier can be necessary through interactions, and replacing a whole sequence is a larger intervention than replacing its final position.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    routes = ("residual_only", "current_position_only", "final_mlp_write_only", "embedding_only", "bus_only")
    x = np.arange(len(routes))
    width = 0.35
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "tinystories")):
        axes[0].bar(x + offset, [results[corpus]["routes"][route]["fidelity"] for route in routes], width, label=corpus)
    axes[0].set_xticks(x, ["residual", "current", "MLP write", "embedding", "value bus"], rotation=20)
    axes[0].set_title("Counterfactual effect fidelity")
    x2 = np.arange(len(COMPONENTS))
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "tinystories")):
        axes[1].bar(x2 + offset, [results[corpus]["shapley_target_projection"][component] for component in COMPONENTS], width, label=corpus)
    axes[1].set_xticks(x2, ["residual", "embedding", "value bus"])
    axes[1].set_title("Signed Shapley target projection")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
