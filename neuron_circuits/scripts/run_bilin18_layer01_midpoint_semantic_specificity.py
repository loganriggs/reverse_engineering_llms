#!/usr/bin/env python3
"""Test whether midpoint circuit compilation is special to semantic edits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_redteam import pair_bootstrap
from run_bilin18_layer01_prospective_tangent_router import (
    cosine, finite_example, jaccard, score_rows, select_indices, tangent_product,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_midpoint_semantic_specificity.json")
OUTPUT = TENSOR_ROOT / "layer01_midpoint_semantic_specificity.json"
REPORT = TENSOR_ROOT / "LAYER01_MIDPOINT_SEMANTIC_SPECIFICITY.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


@torch.no_grad()
def matched_targets(model, families):
    normalized = torch.nn.functional.rms_norm(model.embedding.weight.float(), (model.config.n_embd,))
    target_family = {target: family for family, pairs in families.items() for _, target in pairs}
    candidates = list(target_family)
    result = {}
    for family, pairs in families.items():
        for source, target in pairs:
            true_distance = float((normalized[source] - normalized[target]).norm())
            allowed = [candidate for candidate in candidates if target_family[candidate] != family]
            control = min(allowed, key=lambda candidate: abs(float((normalized[source] - normalized[candidate]).norm()) - true_distance))
            control_distance = float((normalized[source] - normalized[control]).norm())
            result[(source, target)] = {
                "target": control,
                "semantic_distance": true_distance,
                "control_distance": control_distance,
                "relative_mismatch": abs(control_distance - true_distance) / max(true_distance, 1e-30),
            }
    return result


@torch.no_grad()
def evaluate_edit(model, ids, target, layer, edit_position, final_position, rank):
    block = model.blocks[layer]
    decoder = block.mlp.Down.weight.float()
    base, _, full, _ = finite_example(model, ids, target, layer, edit_position, final_position)
    midpoint, _ = tangent_product(model, block, ids, target, layer, edit_position, final_position, path_point=0.5)
    actual_indices = select_indices(score_rows(full, decoder)["local"], rank)
    midpoint_indices = select_indices(score_rows(midpoint, decoder)["local"], rank)
    midpoint_delta = torch.zeros_like(full); midpoint_delta[midpoint_indices] = full[midpoint_indices]
    actual_delta = torch.zeros_like(full); actual_delta[actual_indices] = full[actual_indices]
    effects = downstream_fidelities(model, layer, base, {"full": full, "midpoint": midpoint_delta, "oracle": actual_delta}, final_position)
    return {
        "midpoint": effects["midpoint"],
        "oracle": effects["oracle"],
        "support_jaccard": jaccard(midpoint_indices, actual_indices),
        "product_cosine": cosine(midpoint, full),
    }


@torch.no_grad()
def main():
    config = json.loads(CONFIG.read_text())
    midpoint_config = json.loads(Path(config["source_config"]).read_text())
    redteam = json.loads(Path(midpoint_config["source_config"]).read_text())
    source = json.loads(Path(redteam["source_config"]).read_text())
    data_config = json.loads(Path(source["data_config"]).read_text())
    pair_config = json.loads(Path(source["evaluation_pair_config"]).read_text())
    edit_position, final_position = data_config["edit_position"], data_config["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    controls = matched_targets(model, pair_config["new_families"])
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in data_config["evaluation"].items()
    }
    records = {str(layer): [] for layer in source["layers"]}
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        generator = np.random.default_rng(config["seed"] + corpus_index)
        for family, pairs in pair_config["new_families"].items():
            for source_token, target_token in pairs:
                control = controls[(source_token, target_token)]
                windows = select_windows(rows, source_token, edit_position, final_position, config["contexts_per_pair"], generator)
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64).copy()
                    ids = torch.as_tensor(raw, device="cuda")[None]
                    for layer in source["layers"]:
                        semantic = evaluate_edit(model, ids, target_token, layer, edit_position, final_position, source["rank"])
                        matched = evaluate_edit(model, ids, control["target"], layer, edit_position, final_position, source["rank"])
                        records[str(layer)].append({
                            "corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "context": context_index,
                            "control_target": control["target"], "relative_distance_mismatch": control["relative_mismatch"],
                            **{f"semantic_{key}": value for key, value in semantic.items()},
                            **{f"control_{key}": value for key, value in matched.items()},
                        })
                print("semantic specificity", corpus, family, source_token, target_token, control["target"], flush=True)
    summaries = {}
    for layer in source["layers"]:
        summaries[str(layer)] = {}
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            difference, interval = pair_bootstrap(selected, "semantic_midpoint", "control_midpoint", config["bootstrap_samples"], config["seed"] + layer * 100 + corpus_index)
            summaries[str(layer)][corpus] = {
                key: median([row[key] for row in selected]) for key in (
                    "semantic_midpoint", "semantic_oracle", "semantic_support_jaccard", "semantic_product_cosine",
                    "control_midpoint", "control_oracle", "control_support_jaccard", "control_product_cosine",
                    "relative_distance_mismatch",
                )
            }
            summaries[str(layer)][corpus].update({"semantic_minus_control_pair_median": difference, "semantic_minus_control_pair_ci": interval})
    def row(layer, corpus): return summaries[str(layer)][corpus]
    hypotheses = {
        "S0_semantic_edits_have_better_midpoint_compilation": {"supported": all(row(layer, corpus)["semantic_midpoint"] - row(layer, corpus)["control_midpoint"] >= 0.05 for layer in source["layers"] for corpus in arrays)},
        "S1_semantic_edits_have_more_predictable_support": {"supported": all(row(layer, corpus)["semantic_support_jaccard"] - row(layer, corpus)["control_support_jaccard"] >= 0.05 for layer in source["layers"] for corpus in arrays)},
        "S2_matched_controls_also_compile": {"supported": all(row(layer, corpus)["control_midpoint"] >= 0.80 * row(layer, corpus)["control_oracle"] for layer in source["layers"] for corpus in arrays)},
        "S3_matched_control_direction_is_predictable": {"supported": all(row(layer, corpus)["control_product_cosine"] >= 0.95 for layer in source["layers"] for corpus in arrays)},
        "S4_distance_matching_is_tight": {"supported": all(row(layer, corpus)["relative_distance_mismatch"] <= 0.10 for layer in source["layers"] for corpus in arrays)},
        "S5_any_semantic_advantage_survives_pair_bootstrap": {"supported": all(row(layer, corpus)["semantic_minus_control_pair_ci"][0] > 0 for layer in source["layers"] for corpus in arrays)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "controls": {f"{a}->{b}": value for (a,b),value in controls.items()}, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2))
    lines = ["# Is midpoint circuit compilation semantic-specific?", "", "Every declared semantic edit is paired with a cross-family target chosen to match its normalized embedding-edit distance from the same source token. Both edits use identical contexts, rank, midpoint-JVP budget, realized-delta patch, and output metric.", "", "## Locked hypotheses", ""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Semantic midpoint/oracle/support/cosine | Control midpoint/oracle/support/cosine | Semantic-control pair median (95% CI) | Distance mismatch |", "|---:|---|---:|---:|---:|---:|"]
    for layer in source["layers"]:
        for corpus in arrays:
            item=row(layer,corpus); ci=item["semantic_minus_control_pair_ci"]
            lines.append(f"| {layer} | {corpus} | {item['semantic_midpoint']:.3f}/{item['semantic_oracle']:.3f}/{item['semantic_support_jaccard']:.3f}/{item['semantic_product_cosine']:.3f} | {item['control_midpoint']:.3f}/{item['control_oracle']:.3f}/{item['control_support_jaccard']:.3f}/{item['control_product_cosine']:.3f} | {item['semantic_minus_control_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | {item['relative_distance_mismatch']:.3f} |")
    lines += ["", "If controls compile equally well, midpoint routing is a generic finite-edit property of the tensor architecture. It remains a valid circuit tool but is not evidence that the selected lexical relations share a privileged semantic representation."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__ == "__main__": main()
