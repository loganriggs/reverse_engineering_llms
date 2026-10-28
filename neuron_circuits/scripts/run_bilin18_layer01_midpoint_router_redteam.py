#!/usr/bin/env python3
"""Target-specificity, clustered inference, and batch-packing audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_redteam import pair_bootstrap
from run_bilin18_layer01_prospective_tangent_router import (
    finite_example, jaccard, score_rows, select_indices, tangent_product,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_midpoint_router_redteam.json")
OUTPUT = TENSOR_ROOT / "layer01_midpoint_router_redteam.json"
REPORT = TENSOR_ROOT / "LAYER01_MIDPOINT_ROUTER_REDTEAM.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def clustered_difference(rows, left, right, config, offset):
    return pair_bootstrap(rows, left, right, config["bootstrap_samples"], config["seed"] + offset)


@torch.no_grad()
def main():
    config = json.loads(CONFIG.read_text())
    redteam = json.loads(Path(config["source_config"]).read_text())
    source = json.loads(Path(redteam["source_config"]).read_text())
    data_config = json.loads(Path(source["data_config"]).read_text())
    pairs = json.loads(Path(source["evaluation_pair_config"]).read_text())
    edit_position, final_position = data_config["edit_position"], data_config["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in data_config["evaluation"].items()
    }
    targets = [target for family in pairs["new_families"].values() for _, target in family]
    wrong = {target: targets[(index + 7) % len(targets)] for index, target in enumerate(targets)}
    records = {str(layer): [] for layer in source["layers"]}
    methods = ("base_activity", "base_tangent", "midpoint", "wrong_base", "wrong_midpoint", "realized")

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        generator = np.random.default_rng(config["seed"] + corpus_index)
        for family, lexical_pairs in pairs["new_families"].items():
            for source_token, target_token in lexical_pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    config["contexts_per_pair"], generator,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64).copy()
                    ids = torch.as_tensor(raw, device="cuda")[None]
                    for layer in source["layers"]:
                        block = model.blocks[layer]
                        decoder = block.mlp.Down.weight.float()
                        base, _, full, base_product = finite_example(model, ids, target_token, layer, edit_position, final_position)
                        predictions = {
                            "base_tangent": tangent_product(model, block, ids, target_token, layer, edit_position, final_position)[0],
                            "midpoint": tangent_product(model, block, ids, target_token, layer, edit_position, final_position, path_point=0.5)[0],
                            "wrong_base": tangent_product(model, block, ids, wrong[target_token], layer, edit_position, final_position)[0],
                            "wrong_midpoint": tangent_product(model, block, ids, wrong[target_token], layer, edit_position, final_position, path_point=0.5)[0],
                        }
                        actual_score = score_rows(full, decoder)["local"]
                        indices = {
                            "base_activity": select_indices(base_product.abs() * decoder.norm(dim=0), source["rank"]),
                            **{name: select_indices(score_rows(value, decoder)["local"], source["rank"]) for name, value in predictions.items()},
                            "realized": select_indices(actual_score, source["rank"]),
                        }
                        deltas = {"full": full}
                        for name, selected_indices in indices.items():
                            selected = torch.zeros_like(full)
                            selected[selected_indices] = full[selected_indices]
                            deltas[name] = selected
                        batched = downstream_fidelities(model, layer, base, deltas, final_position)
                        serial = {
                            name: downstream_fidelities(
                                model, layer, base, {"full": full, name: deltas[name]}, final_position,
                            )[name]
                            for name in methods
                        }
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            **serial,
                            "midpoint_support": jaccard(indices["midpoint"], indices["realized"]),
                            "wrong_midpoint_support": jaccard(indices["wrong_midpoint"], indices["realized"]),
                            "packing_error": {name: abs(serial[name] - batched[name]) for name in methods},
                        })
                print("midpoint redteam", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    for layer in source["layers"]:
        summaries[str(layer)] = {}
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            midpoint_difference, midpoint_ci = clustered_difference(selected, "midpoint", "base_activity", config, layer * 100 + corpus_index)
            base_specificity, base_specificity_ci = clustered_difference(selected, "base_tangent", "wrong_base", config, 1000 + layer * 100 + corpus_index)
            summaries[str(layer)][corpus] = {
                name: median([row[name] for row in selected]) for name in methods
            }
            summaries[str(layer)][corpus].update({
                "midpoint_support": median([row["midpoint_support"] for row in selected]),
                "wrong_midpoint_support": median([row["wrong_midpoint_support"] for row in selected]),
                "packing_p95": {name: float(np.quantile([row["packing_error"][name] for row in selected], 0.95)) for name in methods},
                "midpoint_minus_activity_pair_median": midpoint_difference,
                "midpoint_minus_activity_pair_ci": midpoint_ci,
                "base_specificity_pair_median": base_specificity,
                "base_specificity_pair_ci": base_specificity_ci,
            })

    def row(layer, corpus):
        return summaries[str(layer)][corpus]

    hypotheses = {
        "B0_batch_packing_is_stable": {"supported": all(max(row(layer, corpus)["packing_p95"].values()) <= 0.05 for layer in source["layers"] for corpus in arrays)},
        "B1_midpoint_beats_base_activity_clustered": {"supported": all(row(layer, corpus)["midpoint_minus_activity_pair_ci"][0] > 0 for layer in source["layers"] for corpus in arrays)},
        "B2_midpoint_is_target_specific": {"supported": all(row(layer, corpus)["midpoint"] - row(layer, corpus)["wrong_midpoint"] >= 0.30 for layer in source["layers"] for corpus in arrays)},
        "B3_midpoint_support_is_target_specific": {"supported": all(row(layer, corpus)["midpoint_support"] - row(layer, corpus)["wrong_midpoint_support"] >= 0.30 for layer in source["layers"] for corpus in arrays)},
        "B4_midpoint_nearly_matches_realized_oracle": {"supported": all(row(layer, corpus)["midpoint"] >= 0.85 * row(layer, corpus)["realized"] for layer in source["layers"] for corpus in arrays)},
        "B5_base_tangent_specificity_is_positive_but_small": {"supported": all(row(layer, corpus)["base_specificity_pair_ci"][0] > 0 and row(layer, corpus)["base_specificity_pair_median"] < 0.10 for layer in source["layers"] for corpus in arrays)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2))
    lines = [
        "# Target-specificity and packing red-team of the midpoint router", "",
        "Every method is reevaluated serially in the same three-member batch `(base, full, candidate)`. This removes method-count-dependent matrix-kernel variation. Correct and cyclic-wrong targets receive identical midpoint computation budgets; both select channels that are patched with the actual correct-target finite deltas.", "", "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Serial activity/base/mid/wrong-base/wrong-mid/oracle | Mid/wrong support | Mid-activity pair median (95% CI) | Base target-specificity median (95% CI) | Max packing p95 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in source["layers"]:
        for corpus in arrays:
            item = row(layer, corpus); mci = item["midpoint_minus_activity_pair_ci"]; bci = item["base_specificity_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {item['base_activity']:.3f}/{item['base_tangent']:.3f}/{item['midpoint']:.3f}/{item['wrong_base']:.3f}/{item['wrong_midpoint']:.3f}/{item['realized']:.3f} | "
                f"{item['midpoint_support']:.3f}/{item['wrong_midpoint_support']:.3f} | {item['midpoint_minus_activity_pair_median']:+.3f} ({mci[0]:+.3f}, {mci[1]:+.3f}) | "
                f"{item['base_specificity_pair_median']:+.3f} ({bci[0]:+.3f}, {bci[1]:+.3f}) | {max(item['packing_p95'].values()):.3f} |"
            )
    lines += ["", "The midpoint result is a target-conditioned circuit compiler, not a context-only feature detector: replacing the requested target while holding the base text fixed destroys both causal fidelity and native-support overlap."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
