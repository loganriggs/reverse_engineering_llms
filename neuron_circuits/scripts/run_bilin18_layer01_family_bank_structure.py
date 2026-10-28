#!/usr/bin/env python3
"""Test whether broad semantic banks are shared signed codes or unions of pair specialists."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_native_rotation_tournament import (
    coordinate_scores,
    evaluate_deltas,
    selected_native,
    summarize,
)
from run_bilin18_layer3_frozen_channel_transfer import channel_state, jaccard, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_family_bank_structure.json")
OUTPUT = TENSOR_ROOT / "layer01_family_bank_structure.json"
REPORT = TENSOR_ROOT / "LAYER01_FAMILY_BANK_STRUCTURE.md"
FIGURE = TENSOR_ROOT / "layer01_family_bank_structure.png"


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.dot(first, second) / (first.norm() * second.norm()).clamp_min(1e-30))


def recurrence_ranking(rows: torch.Tensor, reference_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    membership = torch.zeros_like(rows, dtype=torch.bool)
    for index, row in enumerate(rows):
        membership[index, top_indices(row.abs(), reference_rank)] = True
    frequency = membership.sum(0)
    mean_abs = rows.abs().mean(0)
    # Frequency is primary.  The bounded tie-breaker cannot cross an integer frequency boundary.
    score = frequency.float() + mean_abs / mean_abs.max().clamp_min(1e-30) * .999
    return score, frequency


def sign_agreement(rows: torch.Tensor, indices: torch.Tensor) -> float:
    signs = torch.sign(rows[:, indices])
    return float((signs.sum(0).abs() / rows.shape[0]).median())


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    frontier_config = json.loads(Path(config["source_config"]).read_text())
    red_config = json.loads(Path(frontier_config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    ranks = config["ranks"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    arrays = {corpus: np.load(spec["path"], mmap_mode="r") for corpus, spec in fixed["evaluation"].items()}
    specs = fixed["evaluation"]

    def reconstruct(record, layer):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)
        return base[layer], counterfactual[layer]

    discovery_saved = [row for row in saved["semantic_records"] if row["corpus"] == config["discovery_corpus"] and row["context_index"] == config["discovery_context_index"]]
    query_saved = [row for row in saved["semantic_records"] if row["context_index"] == config["evaluation_context_index"]]
    family_by_pair = {row["pair"]: row["family"] for row in discovery_saved}
    results = {}

    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        signed = {}
        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            signed[record["pair"]] = coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).cpu()
        pair_names = list(signed)
        supports = {pair: set(top_indices(row.abs(), config["recurrence_reference_rank"]).tolist()) for pair, row in signed.items()}
        within_jaccard, between_jaccard = [], []
        within_cosine, between_cosine = [], []
        for first, second in combinations(pair_names, 2):
            same = family_by_pair[first] == family_by_pair[second]
            (within_jaccard if same else between_jaccard).append(jaccard(supports[first], supports[second]))
            (within_cosine if same else between_cosine).append(cosine(signed[first].abs(), signed[second].abs()))
        geometry = {
            "within_jaccard": float(np.mean(within_jaccard)),
            "between_jaccard": float(np.mean(between_jaccard)),
            "jaccard_advantage": float(np.mean(within_jaccard) - np.mean(between_jaccard)),
            "within_abs_cosine": float(np.mean(within_cosine)),
            "between_abs_cosine": float(np.mean(between_cosine)),
            "cosine_advantage": float(np.mean(within_cosine) - np.mean(between_cosine)),
        }
        cache = {}
        recurrence_fractions = []
        sign_agreements = []
        for pair in pair_names:
            family = family_by_pair[pair]
            members = [candidate for candidate in pair_names if candidate != pair and family_by_pair[candidate] == family]
            rows = torch.stack([signed[candidate] for candidate in members])
            recurrence_score, frequency = recurrence_ranking(rows, config["recurrence_reference_rank"])
            cache[pair] = {}
            for rank in ranks:
                magnitude = top_indices(rows.abs().mean(0), rank).cuda()
                signed_common = top_indices(rows.mean(0).abs(), rank).cuda()
                recurrent = top_indices(recurrence_score, rank).cuda()
                cache[pair][rank] = {"magnitude": magnitude, "signed": signed_common, "recurrent": recurrent}
            recurrent256 = cache[pair][256]["recurrent"].cpu()
            recurrence_fractions.append(float((frequency[recurrent256] >= 2).float().mean()))
            repeated = recurrent256[frequency[recurrent256] >= 2]
            if len(repeated):
                sign_agreements.append(sign_agreement(rows, repeated))

        records = []
        route_names = []
        for record in query_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            deltas = {}
            for rank in ranks:
                for name, indices in cache[record["pair"]][rank].items():
                    deltas[f"{name}_{rank}"] = selected_native(state["delta_product"], indices)
            if not route_names:
                route_names = list(deltas)
            routes = evaluate_deltas(model, layer, base, counterfactual, state, deltas, final_position)
            records.append({"corpus": record["corpus"], "family": record["family"], "pair": record["pair"], "routes": routes})
            print("bank structure", layer, record["corpus"], record["family"], record["pair"], flush=True)
        summaries = {corpus: summarize([row for row in records if row["corpus"] == corpus], route_names) for corpus in fixed["evaluation"]}
        results[str(layer)] = {
            "geometry": geometry,
            "median_recurrence_fraction": float(np.median(recurrence_fractions)),
            "minimum_recurrence_fraction": float(np.min(recurrence_fractions)),
            "median_sign_agreement": float(np.median(sign_agreements)) if sign_agreements else 0.0,
            "summaries": summaries,
            "records": records,
        }

    def value(layer, corpus, route):
        return results[str(layer)]["summaries"][corpus][route]["downstream_fidelity"]

    hypotheses = {
        "C1_same_family_native_support_overlaps_more": {"supported": all(results[str(layer)]["geometry"]["jaccard_advantage"] >= .02 for layer in config["layers"])},
        "C2_same_family_contribution_profiles_are_closer": {"supported": all(results[str(layer)]["geometry"]["cosine_advantage"] >= .05 for layer in config["layers"])},
        "C3_signed_common_bank_matches_magnitude_bank": {"supported": all(value(layer, corpus, "magnitude_256") - value(layer, corpus, "signed_256") <= .05 for layer in config["layers"] for corpus in fixed["evaluation"])},
        "C4_recurrence_bank_matches_magnitude_bank": {"supported": all(value(layer, corpus, "magnitude_256") - value(layer, corpus, "recurrent_256") <= .05 for layer in config["layers"] for corpus in fixed["evaluation"])},
        "C5_family_banks_contain_recurrent_channels": {"supported": all(results[str(layer)]["minimum_recurrence_fraction"] >= .5 for layer in config["layers"])},
        "C6_recurrent_channels_have_consistent_sign": {"supported": all(results[str(layer)]["median_sign_agreement"] >= .67 for layer in config["layers"])},
        "C7_layer1_has_more_support_reuse": {"supported": results["1"]["geometry"]["jaccard_advantage"] - results["0"]["geometry"]["jaccard_advantage"] >= .02},
        "C8_common_banks_transfer_ood": {"supported": all(value(layer, "tinystories", f"{name}_256") >= .75 * value(layer, "fineweb", f"{name}_256") for layer in config["layers"] for name in ("signed", "recurrent"))},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": results}, indent=2))

    lines = [
        "# Are broad Layer-0/1 semantic banks shared codes or unions of specialists?", "",
        "For every held-out lexical pair, three rank-matched native banks are built from the other members of its semantic family: mean absolute contribution (allows heterogeneous signs), absolute signed mean (requires a shared signed effect), and recurrence-first ranking (prefers channels appearing in multiple members' top-256 sets). Each is evaluated through the nonlinear suffix.", "",
        "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Family-bank structure](layer01_family_bank_structure.png)", "",
        "| Layer | Within/between Jaccard | Within/between abs cosine | Recurrent fraction median/min | Sign agreement |", "|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]
        geometry = row["geometry"]
        lines.append(f"| {layer} | {geometry['within_jaccard']:.3f}/{geometry['between_jaccard']:.3f} | {geometry['within_abs_cosine']:.3f}/{geometry['between_abs_cosine']:.3f} | {row['median_recurrence_fraction']:.3f}/{row['minimum_recurrence_fraction']:.3f} | {row['median_sign_agreement']:.3f} |")
    lines += ["", "| Layer | Corpus | Rank | Magnitude union | Signed common | Recurrence first |", "|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            for rank in ranks:
                lines.append(f"| {layer} | {corpus} | {rank} | {value(layer, corpus, f'magnitude_{rank}'):.3f} | {value(layer, corpus, f'signed_{rank}'):.3f} | {value(layer, corpus, f'recurrent_{rank}'):.3f} |")
    lines += ["", "If signed-common and recurrence banks match the magnitude bank, the family uses genuinely shared native gates. If only the magnitude bank works, the apparent family code is better described as a union of pair-specific specialists. Support overlap alone is descriptive; finite downstream fidelity adjudicates the code." ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(4)
    labels = ["L0 Fine", "L0 Tiny", "L1 Fine", "L1 Tiny"]
    cells = [(0, "fineweb"), (0, "tinystories"), (1, "fineweb"), (1, "tinystories")]
    width = .25
    for offset, name in zip((-.25, 0, .25), ("magnitude", "signed", "recurrent")):
        axes[0].bar(x + offset, [value(layer, corpus, f"{name}_256") for layer, corpus in cells], width, label=name)
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("rank-256 downstream fidelity")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    axes[1].bar([-.15, .85], [results["0"]["geometry"]["within_jaccard"], results["1"]["geometry"]["within_jaccard"]], .3, label="within family")
    axes[1].bar([.15, 1.15], [results["0"]["geometry"]["between_jaccard"], results["1"]["geometry"]["between_jaccard"]], .3, label="between family")
    axes[1].set_xticks([0, 1], ["Layer 0", "Layer 1"])
    axes[1].set_ylabel("top-256 support Jaccard")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps(hypotheses, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
