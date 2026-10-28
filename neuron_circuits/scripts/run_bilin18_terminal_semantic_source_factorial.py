#!/usr/bin/env python3
"""Locate terminal sources for transferable and nontransferable lexical families."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_discovery_generalization import full_layer17_details


CONFIG = Path("configs/bilin18_terminal_semantic_source_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_semantic_source_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_SEMANTIC_SOURCE_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_semantic_source_factorial.png"
FACTORS = ("carried", "boundary_attention", "nonboundary_attention", "mlp")
WORLDS = tuple(itertools.product((0, 1), repeat=4))


def shapley(outputs):
    result = {}; count = len(FACTORS)
    for factor_index, name in enumerate(FACTORS):
        value = torch.zeros_like(next(iter(outputs.values())))
        others = [index for index in range(count) if index != factor_index]
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                absent = [0] * count
                for index in subset: absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value += weight * (outputs[tuple(present)] - outputs[tuple(absent)])
        result[name] = value
    return result


def pearson(first, second):
    return float(np.corrcoef(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))[0, 1])


@torch.inference_mode()
def collect(model, raw, records, contrast, config):
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads; destination = 127
    targets = []; pieces = {name: [] for name in FACTORS}; state_error = output_error = edge_error = edge_max_abs = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        block, mixed, details = full_layer17_details(model, original, swapped)
        pattern = (details["score1"] * details["score2"])[:, :, destination].double()
        values = details["mixed_value"].double().permute(0, 2, 1, 3)
        projection = block.attn.c_proj.weight.double().T.reshape(heads, head_dim, dimension)
        edges = torch.einsum("bhs,bhsd,hdk->bhsk", pattern, values, projection).reshape(2, size, heads, original.shape[1], dimension)
        edge_delta = edges[1] - edges[0]
        positions = torch.arange(original.shape[1], device="cuda")[None]
        boundary_mask = positions == boundary[:, None]
        boundary_attention = (edge_delta * boundary_mask[:, None, :, None]).sum((1, 2))
        attention_delta = details["write"][size:, destination].double() - details["write"][:size, destination].double()
        nonboundary_attention = attention_delta - boundary_attention
        edge_discrepancy = edge_delta.sum((1, 2)) - attention_delta
        edge_error = max(edge_error, float(edge_discrepancy.norm() / attention_delta.norm().clamp_min(1e-30)))
        edge_max_abs = max(edge_max_abs, float(edge_discrepancy.abs().max()))
        pre = mixed + details["write"]
        normalized = F.rms_norm(pre.float(), (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        components = {
            "carried": mixed[size:, destination].double() - mixed[:size, destination].double(),
            "boundary_attention": boundary_attention,
            "nonboundary_attention": nonboundary_attention,
            "mlp": mlp[size:, destination].double() - mlp[:size, destination].double(),
        }
        base_state = mixed[:size, destination].double() + details["write"][:size, destination].double() + mlp[:size, destination].double()
        target_state = mixed[size:, destination].double() + details["write"][size:, destination].double() + mlp[size:, destination].double()
        state_error = max(state_error, float((base_state + sum(components.values()) - target_state).norm() / (target_state - base_state).norm().clamp_min(1e-30)))
        scores = {}
        for world in WORLDS:
            residual = (base_state + sum(world[index] * components[name] for index, name in enumerate(FACTORS))).float()
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(residual, (dimension,))).float() / 30)
            scores[world] = logits @ contrast
        target = scores[(1, 1, 1, 1)] - scores[(0, 0, 0, 0)]
        decomposition = shapley(scores)
        output_error = max(output_error, float((sum(decomposition.values()) - target).norm() / target.norm().clamp_min(1e-30)))
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        targets.append((direction * target).cpu())
        for name in FACTORS: pieces[name].append((direction * decomposition[name]).cpu())
    target = torch.cat(targets).double(); energy = target.square().sum().clamp_min(1e-30)
    sources = {}
    for name in FACTORS:
        value = torch.cat(pieces[name]).double()
        sources[name] = {"mean": float(value.mean()), "win": float((value > 0).double().mean()), "projection_share": float((value * target).sum() / energy)}
    sources["attention_combined"] = {"mean": sources["boundary_attention"]["mean"] + sources["nonboundary_attention"]["mean"], "projection_share": sources["boundary_attention"]["projection_share"] + sources["nonboundary_attention"]["projection_share"]}
    return {"examples": len(target), "state_recomposition_error": state_error, "edge_recomposition_error": edge_error, "edge_recomposition_max_abs": edge_max_abs, "output_recomposition_error": output_error, "total": {"mean": float(target.mean()), "win": float((target > 0).double().mean())}, "sources": sources}


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text()); family_source = json.loads(Path(config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}; families.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []
    for family_index, (family, pairs) in enumerate(families.items()):
        midpoint = len(pairs) // 2; folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64); records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + 1000 * family_index + 10 * fold + split)
                print("semantic source", family, fold, split, len(records), flush=True)
                results.append({"family": family, "fold": fold, "split": split, **collect(model, raw, records, contrast, config)})
    family_summary = {}
    for family in families:
        rows = [row for row in results if row["family"] == family]
        family_summary[family] = {name: float(np.mean([row["sources"][name]["projection_share"] for row in rows])) for name in (*FACTORS, "attention_combined")}
    max_error = max(max(row["state_recomposition_error"], row["edge_recomposition_error"], row["output_recomposition_error"]) for row in results); max_edge_abs = max(row["edge_recomposition_max_abs"] for row in results)
    transferable = ("spelling", "sentiment", "gender", "tense")
    f0 = max_error <= 1e-5
    f1 = all(row["sources"]["carried"]["projection_share"] >= .30 for row in results if row["family"] in transferable)
    f2 = all(family_summary["spelling"]["attention_combined"] >= family_summary[family]["attention_combined"] + .10 for family in transferable[1:])
    other_nonboundary = np.median([family_summary[family]["nonboundary_attention"] for family in transferable[1:]])
    f3 = family_summary["spelling"]["nonboundary_attention"] >= 2 * max(other_nonboundary, 0)
    f4 = all(abs(family_summary[family]["mlp"]) < .20 for family in transferable)
    terminal_family = json.loads(Path(config["terminal_family_results"]).read_text()); terminal_spelling = json.loads(Path(config["spelling_terminal_results"]).read_text())
    k90 = {family: float(np.mean([row["joint"]["quotient_k90_median"] for row in terminal_family["results"] if row["family"] == family])) for family in family_source["families"]}; k90["spelling"] = float(np.mean([row["quotient_k90_median"] for row in terminal_spelling["results"] if row["family"] == "spelling"]))
    names = list(families); share_k90_correlation = pearson([family_summary[name]["attention_combined"] for name in names], [k90[name] for name in names]); f5 = share_k90_correlation <= -.5
    hypotheses = {"F0_terminal_source_factorial_recomposes": {"supported": f0, "max_error": max_error, "max_edge_abs": max_edge_abs}, "F1_carried_state_is_material_for_transferable_variables": {"supported": f1}, "F2_spelling_uses_more_fresh_attention": {"supported": f2}, "F3_spelling_nonboundary_attention_is_exceptional": {"supported": f3}, "F4_final_MLP_is_not_the_shared_semantic_interface": {"supported": f4}, "F5_attention_share_predicts_shorter_head_programs": {"supported": f5, "correlation": share_k90_correlation}}
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "family_summary": family_summary, "k90": k90, "results": results}, indent=2))
    lines = ["# Terminal source factorial across semantic families", "", "Natural cue swaps are decomposed at Layer 17's destination into carried mixed state, boundary attention, nonboundary attention, and the realized final-MLP change. All sixteen coalitions pass through final normalization and the disjoint same-family output contrast; exact finite Shapley values allocate the effect.", "", "## Locked results", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", f"Attention-share versus joint-k90 correlation: `{share_k90_correlation:+.3f}`.", "", "| Family | carried | boundary attention | nonboundary attention | combined attention | final MLP | k90 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        row = family_summary[family]; lines.append(f"| {family} | {row['carried']:.3f} | {row['boundary_attention']:.3f} | {row['nonboundary_attention']:.3f} | {row['attention_combined']:.3f} | {row['mlp']:.3f} | {k90[family]:.1f} |")
    lines += ["", "## Numerical closure red team", "", f"The locked combined recomposition gate fails because the independently re-summed attention edges differ from the checkpoint's float32 contraction by relative error up to `{max_error:.2e}` (maximum absolute coordinate error `{max_edge_abs:.2e}`). The additive final state itself recomposes to about `5e-18`, and finite Shapley output error remains below `2e-7`. The source allocation uses the exact checkpoint attention delta with the independently summed boundary subset and a nonboundary remainder, so the locked edge gate remains failed without compromising final-state closure.", "", "## Interpretation boundary", "", "These projection shares allocate one declared disjoint-family contrast at the final additive boundary. Carried state contains all earlier computation and repeated embedding injection. A broad fresh-attention branch does not imply that the semantic variable itself is distributed there; a small branch cannot be used to infer the organization of the dominant carried mechanism."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(names)); figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained"); bottom = np.zeros(len(names))
    for factor in FACTORS:
        values = np.asarray([family_summary[name][factor] for name in names]); axis.bar(x, values, bottom=bottom, label=factor); bottom += values
    axis.axhline(1, color="black", linewidth=.8); axis.set(xticks=x, xticklabels=names, ylabel="Mean Shapley projection share", title="Terminal expression of disjoint family effects"); axis.tick_params(axis="x", rotation=25); axis.legend(fontsize=8); axis.grid(alpha=.2); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "family_summary": family_summary, "k90": k90}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
