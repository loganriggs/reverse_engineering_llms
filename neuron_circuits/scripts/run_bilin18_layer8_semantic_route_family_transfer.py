#!/usr/bin/env python3
"""Leave-one-pair-out test of semantic, generic, and lexical attention routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_router_causal_validation import exact_kl, suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import (
    edge_mask,
    output_with_attention,
    selected_attention_delta,
)
from run_bilin18_layer8_semantic_attention_rank_redteam import all_edge_projections
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_route_family_transfer.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_route_family_transfer.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ROUTE_FAMILY_TRANSFER.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def address(row):
    return int(row[1]), int(row[2]), int(row[3])


def select_route(score_rows, rank):
    edges = score_rows[0]["scores"].keys()
    ranked = sorted(
        edges,
        key=lambda edge: np.mean([abs(row["scores"][edge]) for row in score_rows]),
        reverse=True,
    )
    return ranked[:rank]


def shifted_routes(route, heads):
    return {
        f"shift{shift}": [((head + shift) % heads, destination, source)
                          for head, destination, source in route]
        for shift in range(1, heads)
    }


@torch.inference_mode()
def evaluate_routes(model, layer, base, counterfactual, routes, edit_position, final_position):
    block = model.blocks[layer]
    ranked = all_edge_projections(block, base, counterfactual, edit_position, final_position)
    scores = {address(row): row[0] for row in ranked}
    routes = {**routes, "oracle": [address(row) for row in ranked[:len(next(iter(routes.values())))]]}
    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    indices = {}
    local = {}
    for name, selected in routes.items():
        local[name] = float(sum(scores[edge] for edge in selected))
        delta = selected_attention_delta(
            block, base, counterfactual,
            edge_mask(model.config.n_head, base["output"].shape[1], selected,
                      base["output"].device),
        )
        indices[name] = (len(outputs), len(outputs) + 1)
        outputs += [
            output_with_attention(block, base, base["attention_write"] + delta),
            output_with_attention(block, counterfactual,
                                  counterfactual["attention_write"] - delta),
        ]
        x0s += [base["x0"], counterfactual["x0"]]
        buses += [base["value_bus"], counterfactual["value_bus"]]
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, target, target)
    full_kl = exact_kl(base_logits, counterfactual_logits)
    result = {"full_kl": full_kl, "routes": {}}
    for name, (insert_index, remove_index) in indices.items():
        removal = counterfactual_logits - logits[remove_index]
        fidelity, cosine, _ = fisher_fidelity(base_logits, target, removal)
        result["routes"][name] = {
            "local_projection": local[name],
            "removal_target_projection": fisher_inner(
                base_logits, removal, target) / max(target_energy, 1e-30),
            "removal_fidelity": fidelity,
            "removal_cosine": cosine,
            "removal_kl_ratio": exact_kl(base_logits, logits[remove_index]) / max(full_kl, 1e-30),
        }
    return result


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
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()

    fine_config = fixed["evaluation"]["fineweb"]
    fine_source = np.load(fine_config["path"], mmap_mode="r")
    fine_rows = fine_source[fine_config["start"]:fine_config["start"] + fine_config["rows"]]
    rng = np.random.default_rng(path_config["seed"])
    discovery = []
    for family, pairs in parent["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(fine_rows, source_token, edit_position, final_position,
                                     atlas_config["contexts_per_pair"], rng)
            row_index, start = windows[config["discovery_context_index"]]
            base_raw = np.asarray(fine_rows[row_index, start:start + fixed["sequence_length"]],
                                  dtype=np.int64).copy()
            counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
            base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None],
                                  last_layer=config["layer"])[config["layer"]]
            counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None],
                                     last_layer=config["layer"])[config["layer"]]
            ranked = all_edge_projections(model.blocks[config["layer"]], base, counter,
                                          edit_position, final_position)
            discovery.append({
                "family": family, "pair": f"{source_token}->{target_token}",
                "source_token": source_token, "target_token": target_token,
                "scores": {address(row): float(row[0]) for row in ranked},
            })
            print("discover", family, source_token, target_token, flush=True)

    route_bank = {}
    for heldout in discovery:
        same = [row for row in discovery if row["family"] == heldout["family"]
                and row["pair"] != heldout["pair"]]
        outside = [row for row in discovery if row["family"] != heldout["family"]]
        global_rows = [row for row in discovery if row["pair"] != heldout["pair"]]
        route_bank[heldout["pair"]] = {
            "family": select_route(same, config["rank"]),
            "outside": select_route(outside, config["rank"]),
            "global": select_route(global_rows, config["rank"]),
            "pair": select_route([heldout], config["rank"]),
        }

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                pair_name = f"{source_token}->{target_token}"
                windows = select_windows(rows, source_token, edit_position, final_position,
                                         atlas_config["contexts_per_pair"], rng)
                row_index, start = windows[config["evaluation_context_index"]]
                base_raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]],
                                      dtype=np.int64).copy()
                counter_raw = base_raw.copy(); counter_raw[edit_position] = target_token
                base = forward_layers(model, torch.as_tensor(base_raw, device="cuda")[None],
                                      last_layer=config["layer"])[config["layer"]]
                counter = forward_layers(model, torch.as_tensor(counter_raw, device="cuda")[None],
                                         last_layer=config["layer"])[config["layer"]]
                routes = route_bank[pair_name]
                expanded = dict(routes)
                for route_name in ("family", "global"):
                    for shift_name, shifted in shifted_routes(
                        routes[route_name], model.config.n_head,
                    ).items():
                        expanded[f"{route_name}_{shift_name}"] = shifted
                result = evaluate_routes(model, config["layer"], base, counter, expanded,
                                         edit_position, final_position)
                result.update({"family": family, "pair": pair_name, "row": int(row_index)})
                records.append(result)
                print("evaluate", corpus, family, pair_name, flush=True)
        route_names = ("family", "outside", "global", "pair", "oracle")
        summary_rows = {}
        for name in route_names:
            summary_rows[name] = {
                metric: median([row["routes"][name][metric] for row in records])
                for metric in ("removal_target_projection", "removal_fidelity",
                               "removal_cosine", "removal_kl_ratio")
            }
        for name in ("family", "global"):
            shift_values = []
            for row in records:
                shift_values.append(median([
                    row["routes"][f"{name}_shift{shift}"]["removal_target_projection"]
                    for shift in range(1, model.config.n_head)
                ]))
            summary_rows[f"{name}_shift"] = {
                "removal_target_projection": median(shift_values),
            }
        results[corpus] = {"summary": summary_rows, "records": records}

    f0 = all(results[c]["summary"]["family"]["removal_target_projection"]
             >= results[c]["summary"]["outside"]["removal_target_projection"] + .05
             for c in results)
    f1 = all(abs(results[c]["summary"]["global"]["removal_target_projection"]
                     - results[c]["summary"]["family"]["removal_target_projection"]) <= .05
             and abs(results[c]["summary"]["outside"]["removal_target_projection"]
                     - results[c]["summary"]["family"]["removal_target_projection"]) <= .05
             for c in results)
    f2 = all(results[c]["summary"]["pair"]["removal_target_projection"]
             >= results[c]["summary"]["family"]["removal_target_projection"] + .05
             for c in results)
    f3 = all(
        all(results[c]["summary"][name]["removal_target_projection"] >= .05
            and abs(results[c]["summary"][name]["removal_target_projection"])
            >= 1.5 * abs(results[c]["summary"][f"{name}_shift"]["removal_target_projection"])
            for name in ("family", "global")) for c in results
    )
    f4 = all(results[c]["summary"]["oracle"]["removal_target_projection"]
             >= max(results[c]["summary"][name]["removal_target_projection"]
                    for name in ("family", "outside", "global", "pair")) + .05
             for c in results)
    hypotheses = {
        "F0_semantic_family_route_transfers": {"supported": f0},
        "F1_generic_broadcast_route_explains_transfer": {"supported": f1},
        "F2_lexical_pair_route_is_required": {"supported": f2},
        "F3_cross_pair_routes_are_causal": {"supported": f3},
        "F4_oracle_gap_remains": {"supported": f4},
    }
    # Scores are large and derivable; save only route addresses and causal results.
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses,
        "route_bank": {pair: {name: [list(edge) for edge in route]
                              for name, route in routes.items()}
                       for pair, routes in route_bank.items()},
        "results": results,
    }, indent=2))
    lines = ["# Semantic-family versus generic attention-route transfer", "",
             "Layer-4 edge routes learned from held-out lexical pairs are compared with outside-family, global, pair-specific, oracle, and cyclic-head controls under exact finite counterfactual patching.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Family | Outside family | Global | Pair-specific | Oracle | Family shift | Global shift |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        rows = result["summary"]
        lines.append(
            f"| {corpus} | {rows['family']['removal_target_projection']:.3f} | "
            f"{rows['outside']['removal_target_projection']:.3f} | {rows['global']['removal_target_projection']:.3f} | "
            f"{rows['pair']['removal_target_projection']:.3f} | {rows['oracle']['removal_target_projection']:.3f} | "
            f"{rows['family_shift']['removal_target_projection']:.3f} | {rows['global_shift']['removal_target_projection']:.3f} |"
        )
    lines += ["", "Values are median signed target projections for removal. A generic route can be mechanistically reusable without encoding the shared human semantic category; value content and the current state may supply the object being transported."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
