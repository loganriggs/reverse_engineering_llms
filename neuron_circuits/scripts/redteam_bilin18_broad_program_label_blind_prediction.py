#!/usr/bin/env python3
"""Red-team label-blind head discovery against generic magnitude controls."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from run_bilin18_broad_program_label_blind_prediction import (
    aggregate_label_blind_scores, evaluate_subsets, top_subset,
)
from run_bilin18_broad_regime_construction import length_batches, trace
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_broad_program_label_blind_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_program_label_blind_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_PROGRAM_LABEL_BLIND_REDTEAM.md"


@torch.inference_mode()
def native_activity_batch(model, pairs, layer):
    ids = torch.as_tensor([pair["base_ids"] for pair in pairs], device=model.embedding.weight.device)
    _, traces, _ = trace(model, ids)
    details = traces[layer]["details"]
    heads = model.config.n_head
    head_dim = model.config.n_embd // heads
    pattern = details["pattern"][:, :, -1]
    values = details["mixed_value"].permute(0, 2, 1, 3)
    attended = torch.einsum("bhs,bhsd->bhd", pattern, values)
    projection = model.blocks[layer].attn.c_proj.weight.T.reshape(heads, head_dim, model.config.n_embd)
    writes = torch.einsum("bhd,hdk->bhk", attended, projection).float()
    return writes.square().mean(-1).cpu()


def native_activity_scores(model, pairs, layer, batch_size):
    rows = [native_activity_batch(model, batch, layer) for batch in length_batches(pairs, batch_size)]
    return torch.cat(rows).mean(0).sqrt().numpy()


def cyclic_sources(pairs):
    """Break exact pair identity while retaining the within-family source pool."""
    output = []
    for index, pair in enumerate(pairs):
        source = pairs[(index + 1) % len(pairs)]
        row = copy.copy(pair)
        row["source_ids"] = list(source["source_ids"])
        output.append(row)
    return output


def edited_token_pool(selected, families):
    pool = []
    for family in families:
        for pair in selected[family]["discovery"]:
            pool.extend(source for base, source in zip(pair["base_ids"], pair["source_ids"]) if base != source)
    return sorted(set(pool))


def random_replacements(pairs, pool, seed):
    """Preserve edit positions/count but use unrelated tokens from other edits."""
    rng = np.random.default_rng(seed)
    output = []
    for pair in pairs:
        row = copy.copy(pair)
        source = list(pair["base_ids"])
        for position, (base, original) in enumerate(zip(pair["base_ids"], pair["source_ids"])):
            if base == original:
                continue
            choices = [token for token in pool if token not in (base, original)]
            if not choices:
                choices = [token for token in pool if token != base]
            source[position] = int(rng.choice(choices))
        row["source_ids"] = source
        output.append(row)
    return output


def jaccard(first, second):
    first, second = set(first), set(second)
    return float(len(first & second) / len(first | second))


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    construction = json.loads(Path(config["construction_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    selected = reconstruct_selected(tokenizer, construction)
    families = list(source["results"])
    pool = edited_token_pool(selected, families)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    results = {}; maximum_error = 0.0
    for family_index, family in enumerate(families):
        layer = source["results"][family]["layer"]
        discovery = selected[family]["discovery"]
        heldout = selected[family]["heldout"]
        true_heads = source["results"][family]["heldout"]["input_state_norm"]["heads"]
        native_scores = native_activity_scores(model, discovery, layer, construction["config"]["batch_size"])
        cyclic_scores, error = aggregate_label_blind_scores(
            model, cyclic_sources(discovery), layer, construction["config"]["batch_size"],
        )
        maximum_error = max(maximum_error, error)
        named = {
            "true_pair_delta": true_heads,
            "native_base_activity": top_subset(native_scores, 3),
            "cyclic_family_sources": top_subset(cyclic_scores["state_norm"], 3),
        }
        random_names = []
        for replicate in range(config["random_replacement_replicates"]):
            random_pairs = random_replacements(
                discovery, pool, config["seed"] + 1009 * family_index + replicate,
            )
            random_scores, error = aggregate_label_blind_scores(
                model, random_pairs, layer, construction["config"]["batch_size"],
            )
            maximum_error = max(maximum_error, error)
            name = f"random_replacement_{replicate}"
            named[name] = top_subset(random_scores["state_norm"], 3)
            random_names.append(name)
        evaluated, null = evaluate_subsets(
            model, tokenizer, heldout, layer, named, config["subset_chunk_size"],
            construction["config"]["batch_size"],
        )
        random_fidelity = np.asarray([evaluated[name]["energy_fidelity"] for name in random_names])
        random_percentile = np.asarray([evaluated[name]["subset_percentile"] for name in random_names])
        output_heads = source["results"][family]["heldout"]["label_blind_output_rms"]["heads"]
        results[family] = {
            "layer": layer,
            "true_pair_delta": evaluated["true_pair_delta"],
            "native_base_activity": evaluated["native_base_activity"],
            "cyclic_family_sources": {
                **evaluated["cyclic_family_sources"],
                "jaccard_with_true": jaccard(true_heads, evaluated["cyclic_family_sources"]["heads"]),
            },
            "random_replacements": {
                "replicates": len(random_names),
                "mean_fidelity": float(random_fidelity.mean()),
                "minimum_fidelity": float(random_fidelity.min()),
                "maximum_fidelity": float(random_fidelity.max()),
                "mean_percentile": float(random_percentile.mean()),
                "passing_fraction": float(np.mean((random_fidelity >= config["minimum_energy_fidelity"]) & (random_percentile >= config["minimum_subset_percentile"]))),
                "head_sets": [evaluated[name]["heads"] for name in random_names],
                "mean_jaccard_with_true": float(np.mean([jaccard(true_heads, evaluated[name]["heads"]) for name in random_names])),
            },
            "state_output_top3_jaccard": jaccard(true_heads, output_heads),
            "subset_null": null,
        }
        print(f"generic redteam {family} true={evaluated['true_pair_delta']['energy_fidelity']:.3f} native={evaluated['native_base_activity']['energy_fidelity']:.3f} cyclic={evaluated['cyclic_family_sources']['energy_fidelity']:.3f} random={random_fidelity.mean():.3f}", flush=True)

    mean = lambda key, subkey: float(np.mean([row[key][subkey] for row in results.values()]))
    true_mean = mean("true_pair_delta", "energy_fidelity")
    native_mean = mean("native_base_activity", "energy_fidelity")
    random_mean = float(np.mean([row["random_replacements"]["mean_fidelity"] for row in results.values()]))
    cyclic_pass = [family for family, row in results.items()
                   if row["cyclic_family_sources"]["jaccard_with_true"] >= 2 / 3
                   and row["cyclic_family_sources"]["energy_fidelity"] >= config["minimum_energy_fidelity"]]
    random_pass = [family for family, row in results.items()
                   if row["random_replacements"]["mean_fidelity"] >= config["minimum_energy_fidelity"]
                   and row["random_replacements"]["mean_percentile"] >= config["minimum_subset_percentile"]]
    state_output_overlap = float(np.mean([row["state_output_top3_jaccard"] for row in results.values()]))
    motif = source["motif_enrichment"]
    corrected_p = min(1.0, 2 * min(motif["member_one_sided_p"], motif["component_pair_one_sided_p"]))
    hypotheses = {
        "R0_exact_recomposition": {"supported": maximum_error <= 1e-3, "maximum_head_sum_abs": maximum_error},
        "R1_pair_delta_beats_native_activity": {"supported": true_mean - native_mean >= .05, "true_mean_fidelity": true_mean, "native_mean_fidelity": native_mean, "difference": true_mean - native_mean},
        "R2_family_level_pairing_is_sufficient": {"supported": len(cyclic_pass) >= 4, "passing_families": cyclic_pass},
        "R3_arbitrary_matched_edits_do_not_explain_the_result": {"supported": true_mean - random_mean >= .10 or len(random_pass) < 3, "true_mean_fidelity": true_mean, "random_mean_fidelity": random_mean, "difference": true_mean - random_mean, "random_passing_families": random_pass},
        "R4_suffix_magnitude_adds_distinct_ranking_information": {"supported": state_output_overlap < .5, "mean_top3_jaccard": state_output_overlap},
        "R5_static_motif_enrichment_survives_two-test_correction": {"supported": corrected_p <= .05, "bonferroni_p": corrected_p},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "edited_token_pool": pool, "results": results}, indent=2, allow_nan=False))
    lines = ["# Label-blind broad-program discovery: generic-magnitude red team", "",
             "The positive label-blind result still receives a task-defined counterfactual pair and a semantic-discovery-selected layer. This audit asks whether generic active heads, mismatched sources from the same family, or arbitrary tokens at the same edit positions yield the same program.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | True pair fidelity | Native activity fidelity | Cyclic sources fidelity / Jaccard | Random edits mean fidelity / pct / pass fraction |", "|---|---:|---:|---:|---:|"]
    for family, row in results.items():
        random = row["random_replacements"]
        lines.append(f"| {family} | {row['true_pair_delta']['energy_fidelity']:.3f} | {row['native_base_activity']['energy_fidelity']:.3f} | {row['cyclic_family_sources']['energy_fidelity']:.3f} / {row['cyclic_family_sources']['jaccard_with_true']:.3f} | {random['mean_fidelity']:.3f} / {random['mean_percentile']:.3f} / {random['passing_fraction']:.3f} |")
    lines += ["", "## Guardrail", "",
              "This remains a conditional circuit finder: the behavior family, counterfactual construction, and construction layer are supplied. It tests whether answer labels are needed to select heads, not whether the complete circuit can be discovered from raw text with no behavioral hypothesis. High performance from native or random-edit controls would weaken the claim by showing that generic head magnitude is enough."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
