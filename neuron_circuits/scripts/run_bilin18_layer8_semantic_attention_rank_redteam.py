#!/usr/bin/env python3
"""Rank-resolved red-team of locally selected semantic-attention edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_router_causal_validation import exact_kl, suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import (
    edge_mask,
    median,
    output_with_attention,
    pearson,
    selected_attention_delta,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_attention_rank_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_attention_rank_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ATTENTION_RANK_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_attention_rank_redteam.png"


@torch.inference_mode()
def all_edge_projections(block, base, counterfactual, edit_position, final_position):
    """Return every causal edge and its exact local target projection."""
    destination = slice(edit_position, final_position + 1)
    base_details, counterfactual_details = base["details"], counterfactual["details"]
    base_pattern = base_details["pattern"][0, :, destination].float()
    counterfactual_pattern = counterfactual_details["pattern"][0, :, destination].float()
    base_value = base_details["mixed_value"][0].float().permute(1, 0, 2)
    counterfactual_value = counterfactual_details["mixed_value"][0].float().permute(1, 0, 2)
    edge_delta = (
        counterfactual_pattern[..., None] * counterfactual_value[:, None]
        - base_pattern[..., None] * base_value[:, None]
    )
    target = (
        counterfactual["attention_write"][0, destination].float()
        - base["attention_write"][0, destination].float()
    )
    denominator = target.square().sum().clamp_min(1e-30)
    reads = []
    head_dimension = block.attn.head_dim
    for head in range(block.attn.n_head):
        weight = block.attn.c_proj.weight.float()[
            :, head * head_dimension:(head + 1) * head_dimension
        ]
        reads.append(target @ weight)
    read = torch.stack(reads)
    projections = (edge_delta * read[:, :, None]).sum(-1) / denominator
    rows = []
    for head in range(block.attn.n_head):
        for relative_destination, absolute_destination in enumerate(
            range(edit_position, final_position + 1)
        ):
            for source in range(absolute_destination + 1):
                rows.append((
                    float(projections[head, relative_destination, source]),
                    head,
                    absolute_destination,
                    source,
                ))
    rows.sort(key=lambda row: abs(row[0]), reverse=True)
    return rows


def select_routes(ranked, top_ks, bands, edit_position):
    def edges(rows):
        return [(head, destination, source) for _, head, destination, source in rows]

    routes = {f"top{k}": edges(ranked[:k]) for k in top_ks}
    for start, stop in bands:
        routes[f"band{start + 1}_{stop}"] = edges(ranked[start:stop])
    edited = [row for row in ranked if row[3] == edit_position]
    nonedited = [row for row in ranked if row[3] != edit_position]
    routes["edited_source_top8"] = edges(edited[:8])
    routes["nonedited_source_top8"] = edges(nonedited[:8])
    return routes


@torch.inference_mode()
def analyze_layer(model, layer, base, counterfactual, edit_position, final_position, config):
    block = model.blocks[layer]
    ranked = all_edge_projections(
        block, base, counterfactual, edit_position, final_position,
    )
    routes = select_routes(ranked, config["top_ks"], config["rank_bands"], edit_position)
    local_projection = {
        name: float(sum(
            row[0] for row in ranked
            if (row[1], row[2], row[3]) in set(selected)
        ))
        for name, selected in routes.items()
    }
    deltas = {
        name: selected_attention_delta(
            block,
            base,
            counterfactual,
            edge_mask(
                model.config.n_head, base["output"].shape[1], selected,
                base["output"].device,
            ),
        )
        for name, selected in routes.items()
    }
    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    indices = {}
    for name, delta in deltas.items():
        indices[name] = (len(outputs), len(outputs) + 1)
        outputs += [
            output_with_attention(block, base, base["attention_write"] + delta),
            output_with_attention(
                block, counterfactual, counterfactual["attention_write"] - delta,
            ),
        ]
        x0s += [base["x0"], counterfactual["x0"]]
        buses += [base["value_bus"], counterfactual["value_bus"]]
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    target = counterfactual_logits - base_logits
    energy = fisher_inner(base_logits, target, target)
    full_kl = exact_kl(base_logits, counterfactual_logits)
    result = {}
    for name, (insertion_index, removal_index) in indices.items():
        insertion = logits[insertion_index] - base_logits
        removal = counterfactual_logits - logits[removal_index]
        route = {
            "edge_count": len(routes[name]),
            "local_projection": local_projection[name],
        }
        for direction, effect, changed in (
            ("insertion", insertion, logits[insertion_index]),
            ("removal", removal, logits[removal_index]),
        ):
            fidelity, cosine, _ = fisher_fidelity(base_logits, target, effect)
            route[direction] = {
                "target_projection": fisher_inner(base_logits, effect, target) / max(energy, 1e-30),
                "fisher_fidelity": fidelity,
                "fisher_cosine": cosine,
                "kl_ratio_to_full": exact_kl(base_logits, changed) / max(full_kl, 1e-30),
            }
        result[name] = route
    return {"full_kl": full_kl, "routes": result}


def summarize(records, layers, route_names):
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        selected = [record["layers"][layer_index] for record in records]
        routes = {}
        for name in route_names:
            routes[name] = {
                "edge_count": median([row["routes"][name]["edge_count"] for row in selected]),
                "local_projection": median([row["routes"][name]["local_projection"] for row in selected]),
            }
            for direction in ("insertion", "removal"):
                routes[name][direction] = {
                    key: median([row["routes"][name][direction][key] for row in selected])
                    for key in (
                        "target_projection", "fisher_fidelity", "fisher_cosine",
                        "kl_ratio_to_full",
                    )
                }
        layer_rows.append({"layer": layer, "routes": routes})

    band_names = ["band1_8", "band9_16", "band17_32", "band33_64"]
    local_bands, downstream_bands = [], []
    top64_insertion, top64_removal = [], []
    for record in records:
        for layer_row in record["layers"]:
            for band in band_names:
                local_bands.append(layer_row["routes"][band]["local_projection"])
                downstream_bands.append(
                    layer_row["routes"][band]["removal"]["target_projection"]
                )
            top64_insertion.append(
                layer_row["routes"]["top64"]["insertion"]["target_projection"]
            )
            top64_removal.append(
                layer_row["routes"]["top64"]["removal"]["target_projection"]
            )
    return {
        "layers": layer_rows,
        "correlations": {
            "local_rank_band_to_downstream_removal": pearson(local_bands, downstream_bands),
            "top64_insertion_to_removal": pearson(top64_insertion, top64_removal),
        },
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    causal_config = json.loads(Path(config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    causal = json.loads(Path(config["causal_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    route_names = (
        [f"top{k}" for k in config["top_ks"]]
        + [f"band{start + 1}_{stop}" for start, stop in config["rank_bands"]]
        + ["edited_source_top8", "nonedited_source_top8"]
    )
    results, records_by_corpus = {}, {}

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        frozen_records = causal["records"][corpus]
        frozen_cursor = 0
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                row_index, start = windows[0]
                frozen = frozen_records[frozen_cursor]
                frozen_cursor += 1
                expected_pair = f"{source_token}->{target_token}"
                if int(frozen["row"]) != int(row_index) or frozen["pair"] != expected_pair:
                    raise RuntimeError("rank red-team did not reproduce causal panel")
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
                layer_rows = [
                    analyze_layer(
                        model, layer, base_layers[layer], counterfactual_layers[layer],
                        edit_position, final_position, config,
                    )
                    for layer in config["layers"]
                ]
                records.append({
                    "family": family,
                    "pair": expected_pair,
                    "row": int(row_index),
                    "source_start": int(start),
                    "layers": layer_rows,
                })
                print(corpus, family, expected_pair, flush=True)
        if frozen_cursor != len(frozen_records):
            raise RuntimeError("causal-panel record count mismatch")
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records, config["layers"], route_names)

    layer4 = config["layers"].index(4)
    top8_next8_ratio = {}
    edited_source_fraction = {}
    for corpus in results:
        row = results[corpus]["layers"][layer4]["routes"]
        top8_next8_ratio[corpus] = abs(row["band1_8"]["removal"]["target_projection"]) / max(
            abs(row["band9_16"]["removal"]["target_projection"]), 1e-30,
        )
        edited_source_fraction[corpus] = abs(
            row["edited_source_top8"]["removal"]["target_projection"]
        ) / max(abs(row["top8"]["removal"]["target_projection"]), 1e-30)

    hypotheses = {
        "R1_top8_beats_next8": {
            "supported": all(value >= 1.25 for value in top8_next8_ratio.values()),
            "ratios": top8_next8_ratio,
        },
        "R2_more_local_edges_add_downstream_effect": {
            "supported": all(
                results[corpus]["layers"][layer4]["routes"]["top64"]["removal"]["target_projection"]
                >= results[corpus]["layers"][layer4]["routes"]["top8"]["removal"]["target_projection"]
                for corpus in results
            ),
        },
        "R3_local_rank_bands_predict_downstream_bands": {
            "supported": all(
                results[corpus]["correlations"]["local_rank_band_to_downstream_removal"] >= 0.50
                for corpus in results
            ),
        },
        "R4_edited_source_explains_top8": {
            "supported": all(value >= 0.80 for value in edited_source_fraction.values()),
            "fractions": edited_source_fraction,
        },
        "R5_propagated_sources_are_causal": {
            "supported": any(
                all(
                    results[corpus]["layers"][layer_index]["routes"]["nonedited_source_top8"]["removal"]["target_projection"] >= 0.05
                    for corpus in results
                )
                for layer_index in range(len(config["layers"]))
            ),
        },
        "R6_top64_is_a_major_route": {
            "supported": any(
                all(
                    results[corpus]["layers"][layer_index]["routes"]["top64"]["removal"]["target_projection"] >= 0.25
                    for corpus in results
                )
                for layer_index in range(len(config["layers"]))
            ),
        },
        "R7_larger_route_stabilizes_cross_world_patch": {
            "supported": all(
                results[corpus]["correlations"]["top64_insertion_to_removal"] >= 0.70
                for corpus in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Rank-resolved red-team of semantic attention edges",
        "",
        "The earlier random-source control was too easy because most locally selected edges leave the edited token. This red-team ranks every exact edge by local signed projection, patches nested top-k sets and disjoint rank bands, and separately ranks edges from the edited source and from all other sources. Every patch recomputes the layer MLP and the complete suffix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Rank-resolved semantic attention patches](layer8_semantic_attention_rank_redteam.png)",
        "",
        "| Corpus | Layer | Top-1/2/4/8/16/32/64 removal projection | Edit-source top8 | Non-edit top8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for row in result["layers"]:
            routes = row["routes"]
            values = "/".join(
                f"{routes[f'top{k}']['removal']['target_projection']:.3f}"
                for k in config["top_ks"]
            )
            lines.append(
                f"| {corpus} | {row['layer']} | {values} | "
                f"{routes['edited_source_top8']['removal']['target_projection']:.3f} | "
                f"{routes['nonedited_source_top8']['removal']['target_projection']:.3f} |"
            )
    lines += [
        "",
        "| Corpus | Local-band→downstream correlation | Top64 insertion→removal correlation | Top8/next8 at L4 | Edit-source/global top8 at L4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        lines.append(
            f"| {corpus} | {result['correlations']['local_rank_band_to_downstream_removal']:.3f} | "
            f"{result['correlations']['top64_insertion_to_removal']:.3f} | "
            f"{top8_next8_ratio[corpus]:.3f} | {edited_source_fraction[corpus]:.3f} |"
        )
    lines += [
        "",
        "Rank is defined only by the local attention-write projection. The downstream test is independent of that local ordering but uses the same examples; it evaluates route discovery, not semantic abstraction or context-independent recurrence.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for layer_index, layer in enumerate(config["layers"]):
            axis.plot(
                config["top_ks"],
                [
                    results[corpus]["layers"][layer_index]["routes"][f"top{k}"]["removal"]["target_projection"]
                    for k in config["top_ks"]
                ],
                marker="o",
                label=f"layer {layer}",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(config["top_ks"], config["top_ks"])
        axis.set_title(corpus)
        axis.set_xlabel("locally ranked edges patched")
        axis.set_ylabel("downstream target projection")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
