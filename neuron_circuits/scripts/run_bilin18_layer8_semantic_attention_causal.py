#!/usr/bin/env python3
"""Finite downstream validation of the exact semantic-attention edge atlas.

The atlas ranks individual (QK1)(QK2)V terms by their local projection onto a
counterfactual attention-write delta.  This script tests whether those terms
remain useful after the target layer's MLP and every later transformer block.
It inserts counterfactual edge terms into the base run and removes them from
the counterfactual run.  Random controls preserve every selected edge's head
and destination while resampling only its causal source.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_router_causal_validation import exact_kl, suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_attention_causal.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_causal.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_CAUSAL.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_causal.png"
ROUTES = ("oracle8", "matched_random8", "direct_source", "frozen_head")


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() < 1e-12 or second.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def edge_mask(heads, length, edges, device):
    mask = torch.zeros((1, heads, length, length), dtype=torch.float32, device=device)
    for head, destination, source in edges:
        if not (0 <= head < heads and 0 <= source <= destination < length):
            raise ValueError(f"invalid causal edge {(head, destination, source)}")
        mask[0, head, destination, source] = 1
    return mask


def oracle_edges(atlas_layer, edit_position):
    return [
        (
            int(row["head"]),
            edit_position + int(row["destination_relative"]),
            edit_position + int(row["source_relative"]),
        )
        for row in atlas_layer["top_edges"]
    ]


def matched_random_edges(edges, rng):
    """Preserve oracle heads/destinations while sampling distinct other sources."""
    grouped = {}
    for head, destination, source in edges:
        grouped.setdefault((head, destination), []).append(source)
    sampled = []
    for (head, destination), sources in grouped.items():
        forbidden = set(sources)
        candidates = np.asarray(
            [source for source in range(destination + 1) if source not in forbidden],
            dtype=np.int64,
        )
        if len(candidates) < len(sources):
            raise RuntimeError("not enough matched control sources")
        for source in rng.choice(candidates, size=len(sources), replace=False):
            sampled.append((head, destination, int(source)))
    return sampled


def direct_source_edges(heads, edit_position, final_position):
    return [
        (head, destination, edit_position)
        for head in range(heads)
        for destination in range(edit_position, final_position + 1)
    ]


def frozen_head_edges(head, edit_position, final_position):
    return [
        (head, destination, source)
        for destination in range(edit_position, final_position + 1)
        for source in range(destination + 1)
    ]


@torch.inference_mode()
def selected_attention_delta(block, base, counterfactual, mask):
    """Counterfactual-minus-base write contributed by exactly the masked edges."""
    base_details, counterfactual_details = base["details"], counterfactual["details"]
    base_attended = torch.einsum(
        "bhqk,bkhd->bhqd",
        base_details["pattern"].float() * mask,
        base_details["mixed_value"].float(),
    )
    counterfactual_attended = torch.einsum(
        "bhqk,bkhd->bhqd",
        counterfactual_details["pattern"].float() * mask,
        counterfactual_details["mixed_value"].float(),
    )
    delta = counterfactual_attended - base_attended
    batch, heads, length, head_dimension = delta.shape
    flattened = delta.transpose(1, 2).contiguous().view(batch, length, heads * head_dimension)
    return F.linear(flattened, block.attn.c_proj.weight.float(), bias=None)


@torch.inference_mode()
def output_with_attention(block, record, attention_write):
    dimension = attention_write.shape[-1]
    pre = record["mixed"] + attention_write
    normalized = F.rms_norm(pre, (dimension,))
    product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
    mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
    return pre + mlp_write


def route_metrics(reference_logits, target, effect, changed_logits, full_kl):
    target_energy = fisher_inner(reference_logits, target, target)
    projection = fisher_inner(reference_logits, effect, target) / max(target_energy, 1e-30)
    fidelity, cosine, _ = fisher_fidelity(reference_logits, target, effect)
    return {
        "target_projection": projection,
        "fisher_fidelity": fidelity,
        "fisher_cosine": cosine,
        "kl": exact_kl(reference_logits, changed_logits),
        "kl_ratio_to_full": exact_kl(reference_logits, changed_logits) / max(full_kl, 1e-30),
    }


@torch.inference_mode()
def analyze_layer(model, layer, base, counterfactual, atlas_layer, frozen_head, edit_position,
                  final_position, random_generator):
    block = model.blocks[layer]
    heads = model.config.n_head
    length = base["output"].shape[1]
    oracle = oracle_edges(atlas_layer, edit_position)
    routes = {
        "oracle8": oracle,
        "matched_random8": matched_random_edges(oracle, random_generator),
        "direct_source": direct_source_edges(heads, edit_position, final_position),
        "frozen_head": frozen_head_edges(frozen_head, edit_position, final_position),
    }

    route_deltas = {}
    route_local_projection = {}
    local_target = (
        counterfactual["attention_write"][:, edit_position:final_position + 1].float()
        - base["attention_write"][:, edit_position:final_position + 1].float()
    )
    local_energy = float(local_target.square().sum().clamp_min(1e-30))
    for name, edges in routes.items():
        mask = edge_mask(heads, length, edges, base["output"].device)
        delta = selected_attention_delta(block, base, counterfactual, mask)
        route_deltas[name] = delta
        route_local_projection[name] = float(
            (delta[:, edit_position:final_position + 1] * local_target).sum() / local_energy
        )

    layer_outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    indices = {}
    for name in ROUTES:
        delta = route_deltas[name]
        insertion = output_with_attention(block, base, base["attention_write"] + delta)
        removal = output_with_attention(
            block, counterfactual, counterfactual["attention_write"] - delta,
        )
        indices[name] = (len(layer_outputs), len(layer_outputs) + 1)
        layer_outputs += [insertion, removal]
        x0s += [base["x0"], counterfactual["x0"]]
        buses += [base["value_bus"], counterfactual["value_bus"]]

    logits = suffix_from_layer(
        model,
        layer,
        torch.cat(x0s),
        torch.cat(buses),
        torch.cat(layer_outputs),
        final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    target = counterfactual_logits - base_logits
    full_kl = exact_kl(base_logits, counterfactual_logits)
    result = {}
    for name, (insertion_index, removal_index) in indices.items():
        insertion_effect = logits[insertion_index] - base_logits
        removal_effect = counterfactual_logits - logits[removal_index]
        result[name] = {
            "edge_count": len(routes[name]),
            "local_target_projection": route_local_projection[name],
            "insertion": route_metrics(
                base_logits, target, insertion_effect, logits[insertion_index], full_kl,
            ),
            "removal": route_metrics(
                base_logits, target, removal_effect, logits[removal_index], full_kl,
            ),
        }
    atlas_projection = float(sum(row["projection"] for row in atlas_layer["top_edges"]))
    return {
        "full_kl": full_kl,
        "base_logits": base_logits,
        "counterfactual_logits": counterfactual_logits,
        "atlas_oracle_local_projection": atlas_projection,
        "oracle_local_recomposition_error": abs(
            result["oracle8"]["local_target_projection"] - atlas_projection
        ),
        "routes": result,
    }


def summarize(records):
    layers = []
    for layer in range(9):
        selected = [record["layers"][layer] for record in records]
        route_summary = {}
        for route in ROUTES:
            route_summary[route] = {
                "edge_count": median([row["routes"][route]["edge_count"] for row in selected]),
                "local_target_projection": median([
                    row["routes"][route]["local_target_projection"] for row in selected
                ]),
            }
            for direction in ("insertion", "removal"):
                route_summary[route][direction] = {
                    key: median([row["routes"][route][direction][key] for row in selected])
                    for key in (
                        "target_projection", "fisher_fidelity", "fisher_cosine",
                        "kl", "kl_ratio_to_full",
                    )
                }
        layers.append({"layer": layer, "routes": route_summary})
    oracle_local = [
        row["layers"][layer]["atlas_oracle_local_projection"]
        for row in records for layer in range(9)
    ]
    insertion = [
        row["layers"][layer]["routes"]["oracle8"]["insertion"]["target_projection"]
        for row in records for layer in range(9)
    ]
    removal = [
        row["layers"][layer]["routes"]["oracle8"]["removal"]["target_projection"]
        for row in records for layer in range(9)
    ]
    return {
        "layers": layers,
        "correlations": {
            "local_oracle_to_downstream_removal": pearson(oracle_local, removal),
            "local_oracle_to_downstream_insertion": pearson(oracle_local, insertion),
            "downstream_insertion_to_removal": pearson(insertion, removal),
        },
    }


def same_layer_gate(results, route, threshold):
    return any(
        all(
            max(
                results[corpus]["layers"][layer]["routes"][route]["insertion"]["target_projection"],
                results[corpus]["layers"][layer]["routes"][route]["removal"]["target_projection"],
            ) >= threshold
            for corpus in results
        )
        for layer in range(9)
    )


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    atlas_config = json.loads(Path(config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    atlas = json.loads(Path(config["atlas_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    frozen_heads = [
        int(row["most_common_top_head"])
        for row in atlas["results"]["fineweb"]["layers"]
    ]
    results = {}
    records_by_corpus = {}
    maximum_local_error = 0.0
    maximum_native_recomposition_error = 0.0

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        # Recreate the atlas contexts exactly; it selected two windows per pair.
        context_rng = np.random.default_rng(path_config["seed"] + corpus_index)
        control_rng = np.random.default_rng(config["seed"] + corpus_index)
        atlas_records = atlas["results"][corpus]["records"]
        atlas_cursor = 0
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], context_rng,
                )
                pair_atlas = atlas_records[
                    atlas_cursor:atlas_cursor + atlas_config["contexts_per_pair"]
                ]
                atlas_cursor += atlas_config["contexts_per_pair"]
                for context_index, ((row_index, start), atlas_record) in enumerate(
                    zip(windows, pair_atlas)
                ):
                    if context_index >= config["contexts_per_pair"]:
                        continue
                    expected_pair = f"{source_token}->{target_token}"
                    if (
                        atlas_record["family"] != family
                        or atlas_record["pair"] != expected_pair
                        or int(atlas_record["row"]) != int(row_index)
                    ):
                        raise RuntimeError("failed to reproduce the frozen atlas panel")
                    base_raw = np.asarray(
                        rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                    ).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[edit_position] = target_token
                    base_layers = forward_layers(
                        model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=8,
                    )
                    counterfactual_layers = forward_layers(
                        model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=8,
                    )
                    layer_rows = []
                    example_reference = None
                    for layer in range(9):
                        layer_row = analyze_layer(
                            model, layer, base_layers[layer], counterfactual_layers[layer],
                            atlas_record["layers"][layer], frozen_heads[layer], edit_position,
                            final_position, control_rng,
                        )
                        maximum_local_error = max(
                            maximum_local_error, layer_row["oracle_local_recomposition_error"],
                        )
                        if example_reference is None:
                            example_reference = layer_row["base_logits"]
                        maximum_native_recomposition_error = max(
                            maximum_native_recomposition_error,
                            float((layer_row["base_logits"] - example_reference).abs().max()),
                        )
                        layer_row.pop("base_logits")
                        layer_row.pop("counterfactual_logits")
                        layer_rows.append(layer_row)
                    records.append({
                        "family": family,
                        "pair": expected_pair,
                        "row": int(row_index),
                        "source_start": int(start),
                        "layers": layer_rows,
                    })
                    print(corpus, family, expected_pair, flush=True)
        if atlas_cursor != len(atlas_records):
            raise RuntimeError("atlas record count did not match recreated panel")
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records)

    # FineWeb is the discovery corpus: select one layer there, then require its
    # oracle/random advantage to transfer to TinyStories without re-selection.
    discovery_layer = max(
        range(9),
        key=lambda layer: max(
            abs(results["fineweb"]["layers"][layer]["routes"]["oracle8"][direction]["target_projection"])
            for direction in ("insertion", "removal")
        ),
    )
    oracle_random_ratios = {}
    for corpus in results:
        oracle = max(
            abs(results[corpus]["layers"][discovery_layer]["routes"]["oracle8"][direction]["target_projection"])
            for direction in ("insertion", "removal")
        )
        random = max(
            abs(results[corpus]["layers"][discovery_layer]["routes"]["matched_random8"][direction]["target_projection"])
            for direction in ("insertion", "removal")
        )
        oracle_random_ratios[corpus] = oracle / max(random, 1e-30)

    hypotheses = {
        "L1_sparse_oracle_edges_are_downstream_causal": {
            "supported": same_layer_gate(results, "oracle8", 0.10),
        },
        "L2_sparse_edges_beat_random_controls": {
            "supported": all(value >= 2 for value in oracle_random_ratios.values()),
            "fineweb_selected_layer": discovery_layer,
            "oracle_to_random_ratios": oracle_random_ratios,
        },
        "L3_frozen_recurrent_heads_are_downstream_causal": {
            "supported": same_layer_gate(results, "frozen_head", 0.10),
        },
        "L4_direct_source_edges_are_a_major_downstream_route": {
            "supported": same_layer_gate(results, "direct_source", 0.20),
        },
        "L5_local_edge_score_predicts_downstream_effect": {
            "supported": all(
                results[corpus]["correlations"]["local_oracle_to_downstream_removal"] >= 0.50
                for corpus in results
            ),
        },
        "L6_insertion_and_removal_agree": {
            "supported": all(
                results[corpus]["correlations"]["downstream_insertion_to_removal"] >= 0.70
                for corpus in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "frozen_fineweb_heads": frozen_heads,
        "fineweb_selected_layer": discovery_layer,
        "hypotheses": hypotheses,
        "maximum_oracle_local_recomposition_error": maximum_local_error,
        "maximum_cross_layer_native_logit_error": maximum_native_recomposition_error,
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Finite causal validation of semantic attention edges",
        "",
        "The exact `(QK1)(QK2)V` terms ranked by the local attention atlas are now replaced across counterfactual worlds, the target layer's bilinear MLP is recomputed, and every later block is run normally. `oracle8` uses the eight locally strongest signed-projection edges. Each `matched_random8` edge preserves its oracle head and destination but resamples a different causal source. `direct_source` replaces every edge leaving the edited position; `frozen_head` replaces every affected edge in the head selected only from the FineWeb atlas.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Finite semantic edge patches](layer8_semantic_attention_causal.png)",
        "",
        "| Corpus | Layer | Oracle-8 insertion/removal | Matched random insertion/removal | Direct-source insertion/removal | Frozen-head insertion/removal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for row in result["layers"]:
            values = []
            for route in ROUTES:
                values.append(
                    f"{row['routes'][route]['insertion']['target_projection']:.3f} / "
                    f"{row['routes'][route]['removal']['target_projection']:.3f}"
                )
            lines.append(
                f"| {corpus} | {row['layer']} | {values[0]} | {values[1]} | "
                f"{values[2]} | {values[3]} |"
            )
    lines += [
        "",
        "| Corpus | Local→removal correlation | Local→insertion correlation | Insertion→removal correlation |",
        "|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        corr = result["correlations"]
        lines.append(
            f"| {corpus} | {corr['local_oracle_to_downstream_removal']:.3f} | "
            f"{corr['local_oracle_to_downstream_insertion']:.3f} | "
            f"{corr['downstream_insertion_to_removal']:.3f} |"
        )
    lines += [
        "",
        f"FineWeb selected layer: **{discovery_layer}**. Oracle/random absolute-projection ratios: "
        + ", ".join(f"{corpus}={value:.2f}" for corpus, value in oracle_random_ratios.items()) + ".",
        f"Maximum atlas/direct local projection discrepancy: `{maximum_local_error:.2e}`. "
        f"Maximum base-logit discrepancy across split points: `{maximum_native_recomposition_error:.2e}`.",
        "",
        "An edge patch is a cross-world causal replacement, not a claim that the edge independently stores the edited word's meaning. Large source or whole-head patches test routes with very different budgets; only oracle-eight versus matched-random-eight is budget matched.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        layers = np.arange(9)
        for route, marker in (("oracle8", "o"), ("matched_random8", "x"),
                              ("direct_source", "s"), ("frozen_head", "^")):
            axis.plot(
                layers,
                [
                    results[corpus]["layers"][layer]["routes"][route]["removal"]["target_projection"]
                    for layer in layers
                ],
                marker=marker,
                label=route,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(corpus)
        axis.set_xlabel("patched layer")
        axis.set_ylabel("downstream target projection")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
