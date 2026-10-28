#!/usr/bin/env python3
"""Move a token edit into the past and retest exact squared-attention structure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import (
    FACTORS,
    attention_world_writes,
    logits_from_attention_writes,
    trace,
    vector_shapley,
)
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_attention_source_distance_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_attention_source_distance_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_ATTENTION_SOURCE_DISTANCE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_attention_source_distance_redteam.png"


def sequence_head_writes(block, base_attended, changed_attended):
    delta = changed_attended - base_attended  # B,H,Q,Dh
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    writes = []
    for head in range(heads):
        weight = block.attn.c_proj.weight[:, head * head_dimension:(head + 1) * head_dimension]
        writes.append(delta[:, head] @ weight.T)
    return torch.stack(writes)  # H,B,Q,D


def empty_accumulator(heads):
    return {
        "examples": 0, "full_kl": 0.0, "insert_kl": 0.0,
        "local_den": 0.0, "local_cross": {factor: 0.0 for factor in FACTORS},
        "insert_den": 0.0, "insert_cross": {factor: 0.0 for factor in FACTORS},
        "remove_den": 0.0, "remove_cross": {factor: 0.0 for factor in FACTORS},
        "head_local_cross": [0.0] * heads, "head_insert_cross": [0.0] * heads,
        "head_insert_den": 0.0, "head_sum_error": 0.0,
    }


def update(acc, block, base, changed, writes, insert_logits, remove_logits, head_logits, native_base, native_changed, edit_position):
    local_worlds = writes[:, :, edit_position:]
    local_target = local_worlds[15] - local_worlds[0]
    local_den = local_target.square().sum().clamp_min(1e-30)
    acc["local_den"] += float(local_den)
    for factor, effect in vector_shapley(local_worlds).items():
        acc["local_cross"][factor] += float((effect * local_target).sum())
    for prefix, logits in (("insert", insert_logits), ("remove", remove_logits)):
        target = logits[15] - logits[0]; reference = logits[0]
        energy, _, _, _ = fisher_terms(reference, target, target)
        acc[f"{prefix}_den"] += float(energy.sum())
        for factor, effect in vector_shapley(logits).items():
            _, _, cross, _ = fisher_terms(reference, target, effect)
            acc[f"{prefix}_cross"][factor] += float(cross.sum())
    head_effects = head_logits[1:] - head_logits[0]
    insertion_target = insert_logits[15] - insert_logits[0]
    insertion_energy, _, _, _ = fisher_terms(insert_logits[0], insertion_target, insertion_target)
    _, _, _, sum_error = fisher_terms(insert_logits[0], insertion_target, head_effects.sum(0))
    acc["head_insert_den"] += float(insertion_energy.sum()); acc["head_sum_error"] += float(sum_error.sum())
    head_writes = sequence_head_writes(block, base["attended"], changed["attended"])[:, :, edit_position:]
    for head in range(block.attn.n_head):
        acc["head_local_cross"][head] += float((head_writes[head] * local_target).sum())
        _, _, cross, _ = fisher_terms(insert_logits[0], insertion_target, head_effects[head])
        acc["head_insert_cross"][head] += float(cross.sum())
    acc["examples"] += len(native_base)
    acc["full_kl"] += float(exact_kl(native_base, native_changed).sum())
    acc["insert_kl"] += float(exact_kl(insert_logits[0], insert_logits[15]).sum())


def finalize(acc):
    local = {factor: acc["local_cross"][factor] / max(acc["local_den"], 1e-30) for factor in FACTORS}
    insertion = {factor: acc["insert_cross"][factor] / max(acc["insert_den"], 1e-30) for factor in FACTORS}
    removal = {factor: acc["remove_cross"][factor] / max(acc["remove_den"], 1e-30) for factor in FACTORS}
    local_heads = [value / max(acc["local_den"], 1e-30) for value in acc["head_local_cross"]]
    insert_heads = [value / max(acc["insert_den"], 1e-30) for value in acc["head_insert_cross"]]
    return {
        "examples": acc["examples"], "mean_full_token_kl": acc["full_kl"] / max(acc["examples"], 1),
        "attention_to_full_kl": acc["insert_kl"] / max(acc["full_kl"], 1e-30),
        "local_write_shapley": local, "insertion_causal_shapley": insertion, "removal_causal_shapley": removal,
        "local_head_projection": local_heads, "insertion_head_projection": insert_heads,
        "top_local_head": int(np.argmax(np.abs(local_heads))),
        "top_local_head_absolute_share": max(abs(v) for v in local_heads) / max(sum(abs(v) for v in local_heads), 1e-30),
        "isolated_head_sum_fidelity": 1 - acc["head_sum_error"] / max(acc["head_insert_den"], 1e-30),
    }


@torch.inference_mode()
def evaluate_position(model, tokenizer, rows, edit_position, config):
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    accumulators = {layer: empty_accumulator(model.config.n_head) for layer in config["layers"]}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda"); keep = base_ids[:, edit_position] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline
        base_rows, native_base, _ = trace(model, base_ids); changed_rows, native_changed, _ = trace(model, changed_ids)
        for layer in config["layers"]:
            block = model.blocks[layer]; base = base_rows[layer]; changed = changed_rows[layer]
            writes, _ = attention_world_writes(block, base["details"], changed["details"])
            insert_logits = logits_from_attention_writes(model, layer, base, writes)
            remove_logits = logits_from_attention_writes(model, layer, changed, writes)
            head_deltas = sequence_head_writes(block, base["details"]["attended"], changed["details"]["attended"])
            head_worlds = [base["details"]["write"]]
            for head in range(model.config.n_head): head_worlds.append(base["details"]["write"] + head_deltas[head])
            head_logits = logits_from_attention_writes(model, layer, base, torch.stack(head_worlds))
            update(accumulators[layer], block, base["details"], changed["details"], writes, insert_logits, remove_logits, head_logits, native_base, native_changed, edit_position)
    return {str(layer): finalize(accumulators[layer]) for layer in config["layers"]}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    results = {}
    for corpus, spec in config["evaluation"].items():
        rows = load_rows(spec); results[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluating", corpus, "position", position, flush=True)
            results[corpus][str(position)] = evaluate_position(model, tokenizer, rows, position, config)
    best_layers = {corpus: {str(position): max(config["layers"], key=lambda layer: results[corpus][str(position)][str(layer)]["attention_to_full_kl"]) for position in config["edit_positions"]} for corpus in results}
    hypotheses = {
        "S1_earlier_edits_change_final_behavior": {"supported": all(results[c][str(p)][str(config["layers"][0])]["mean_full_token_kl"] >= .001 for c in results for p in config["edit_positions"][:-1])},
        "S2_attention_is_material_for_earlier_edits": {"supported": all(max(results[c][str(p)][str(layer)]["attention_to_full_kl"] for layer in config["layers"]) >= .10 for c in results for p in config["edit_positions"][:-1])},
        "S3_earlier_edits_depend_more_on_attention": {"supported": all(np.mean([max(results[c][str(p)][str(layer)]["attention_to_full_kl"] for layer in config["layers"]) for p in config["edit_positions"][:-1]]) > max(results[c][str(config["edit_positions"][-1])][str(layer)]["attention_to_full_kl"] for layer in config["layers"]) for c in results)},
        "S4_source_key_role_grows_with_distance": {"supported": all(max(abs(results[c]["63"][str(layer)]["local_write_shapley"]["key"]) for layer in config["layers"]) > max(abs(results[c]["127"][str(layer)]["local_write_shapley"]["key"]) for layer in config["layers"]) for c in results)},
        "S5_persistent_value_bus_is_used_for_long_range_edits": {"supported": all(max(abs(results[c]["63"][str(layer)]["local_write_shapley"]["value_bus"]) for layer in config["layers"]) >= .10 for c in results)},
        "S6_material_attention_depth_transfers_ood": {"supported": all(max(best_layers[c][str(p)] for c in results) - min(best_layers[c][str(p)] for c in results) <= 2 for p in config["edit_positions"]), "best_layers": best_layers},
        "S7_material_attention_has_a_stable_factor": {"supported": all((lambda row: max(FACTORS, key=lambda f: abs(row["insertion_causal_shapley"][f])) == max(FACTORS, key=lambda f: abs(row["removal_causal_shapley"][f])))(results[c][str(p)][str(best_layers[c][str(p)])]) for c in results for p in config["edit_positions"])},
        "S8_head_additivity_survives_long_range_routing": {"supported": all(results[c][str(p)][str(best_layers[c][str(p)])]["isolated_head_sum_fidelity"] >= .75 for c in results for p in config["edit_positions"][:-1])},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "best_layers": best_layers, "results": results}, indent=2, allow_nan=False))

    positions = config["edit_positions"]; distances = [config["prediction_position"] - p for p in positions]; fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for corpus in results:
        axes[0].plot(distances, [max(results[corpus][str(p)][str(layer)]["attention_to_full_kl"] for layer in config["layers"]) for p in positions], marker="o", label=corpus)
        axes[1].plot(distances, [best_layers[corpus][str(p)] for p in positions], marker="o", label=corpus)
        axes[2].plot(distances, [max(abs(results[corpus][str(p)][str(layer)]["local_write_shapley"]["key"]) for layer in config["layers"]) for p in positions], marker="o", label=corpus)
    axes[0].axhline(.1, color="black", ls="--", lw=.8); axes[0].set_title("Best attention-only KL / full KL"); axes[1].set_title("Most material attention layer"); axes[2].set_title("Maximum local key allocation")
    for axis in axes: axis.set_xlabel("distance from prediction"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = ["# Source-distance red team for squared attention", "", "The ordinary-token-to-newline edit is moved 8, 32, and 64 positions before the final prediction, with the original final-position edit retained as a reference. Q/K/current-value/value-bus hybrids replace the entire changed attention-write sequence at each layer, and native head deltas are patched at every affected destination.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Source distance](bilin12_attention_source_distance_redteam.png)", "", "| Corpus | Edit position / distance | Full KL | Best layer | Best attention/full KL | Best-layer Q / K / current-V / bus | Top head/share | Head-sum fidelity |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for position, distance in zip(positions, distances):
            layer = best_layers[corpus][str(position)]; row = results[corpus][str(position)][str(layer)]; factors = " / ".join(f"{row['insertion_causal_shapley'][f]:.3f}" for f in FACTORS)
            lines.append(f"| {corpus} | {position} / {distance} | {row['mean_full_token_kl']:.4f} | {layer} | {row['attention_to_full_kl']:.3f} | {factors} | {row['top_local_head']} / {row['top_local_head_absolute_share']:.3f} | {row['isolated_head_sum_fidelity']:.3f} |")
    lines += ["", "A single-layer attention insertion is not a percentage of information stored. For earlier edits, however, cross-position attention is structurally required somewhere in the full path. Failure of a single layer to reproduce the full edit can therefore indicate distributed routing across layers rather than an attention-free mechanism."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
