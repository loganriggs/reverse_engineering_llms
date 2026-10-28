#!/usr/bin/env python3
"""Swap the states used by exact local suffix operators across contexts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_layers0_3_prospective_chart_retrieval import current_state_signatures, unit_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_local_operator_state_swap.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_local_operator_state_swap.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_LOCAL_OPERATOR_STATE_SWAP.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_local_operator_state_swap.png"


def source_for(spec, family, layer):
    if family == "mlp" and spec.get("mlp_source"):
        return spec["mlp_source"]
    if family == "attention" and spec.get("attention_source"):
        return spec["attention_source"]
    return "target" if layer in spec.get("exact_layers", set()) else spec.get("default_source", "nearest")


def hybrid_program(model, traces, source_layer, position, contrast, specs):
    names = list(specs)
    current = final_adjoint(model, traces["target"][-1]["output"], position, contrast).repeat(len(names), 1, 1, 1)
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        mlp_rows = torch.zeros_like(current)
        for source in ("target", "nearest", "farthest"):
            selected = [row for row, name in enumerate(names) if source_for(specs[name], "mlp", layer) == source]
            if selected:
                mlp_rows[selected] = mlp_vjp(block, traces[source][layer]["pre"].float(), current[selected])
        updated = current + mlp_rows
        attention_rows = torch.zeros_like(current)
        for source in ("target", "nearest", "farthest"):
            selected = [row for row, name in enumerate(names) if source_for(specs[name], "attention", layer) == source]
            if selected:
                trace = traces[source]
                previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
                mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
                bus = trace[layer - 1]["value_bus"]
                attention_rows[selected] = attention_vjp(block, mixed, bus, updated[selected])
        current = block.lambdas[0].float() * (updated + attention_rows)
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, position] @ down for row, name in enumerate(names)}


def specifications(source_layer, block_count, near_widths):
    suffix = list(range(source_layer + 1, block_count))
    result = {
        "exact_all": {"default_source": "target"},
        "nearest_all": {"default_source": "nearest"},
        "farthest_all": {"default_source": "farthest"},
        "exact_mlp": {"mlp_source": "target", "attention_source": "nearest"},
        "exact_attention": {"mlp_source": "nearest", "attention_source": "target"},
        # Post-hoc poisoning controls added after the first state-swap run.
        "exact_mlp_only": {"mlp_source": "target", "attention_source": "zero"},
        "exact_attention_only": {"mlp_source": "zero", "attention_source": "target"},
        "nearest_mlp_only": {"mlp_source": "nearest", "attention_source": "zero"},
        "nearest_attention_only": {"mlp_source": "zero", "attention_source": "nearest"},
        "residual_only": {"mlp_source": "zero", "attention_source": "zero"},
    }
    for width in near_widths:
        result[f"exact_near_{width}"] = {"default_source": "nearest", "exact_layers": set(suffix[:width])}
        result[f"exact_late_{width}"] = {"default_source": "nearest", "exact_layers": set(suffix[-width:])}
    return result


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]
    values = [row["metrics"][method] for row in rows]; shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])),
        "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)),
        "gradient_relative_error_median": float(np.median([value["gradient_relative_error"] for value in values])),
        "contribution_cosine_median": float(np.median([value["contribution_cosine"] for value in values])),
        "true_leverage_ratio_median": float(np.median([value["true_leverage_ratio"] for value in values])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
    }


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); discovery = corpus[source["start_row"]:source["start_row"] + source["discovery_rows"]]; heldout_start = source["start_row"] + source["discovery_rows"]
    atlas = np.load(config["atlas_arrays"])
    selected = []
    for feature_name in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature_name]
        selected.extend((next(row for row in rows if row["selection"] == "extreme"), next(row for row in rows if row["selection"] == "uniform")))
    records = []
    for site_index, row in enumerate(selected):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]; position = row["recipient_position"]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); target_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        reference_cache = {}
        for source_layer in config["source_layers"]:
            bank_positions = atlas[f"layer{source_layer}_position"]; candidates = np.flatnonzero(bank_positions == position)
            bank_signature = atlas[f"layer{source_layer}_path"][candidates].astype(np.float32)
            with torch.no_grad(): query = current_state_signatures(model, target_trace, recipient_ids, source_layer, position)["path"][0].cpu().numpy().astype(np.float32)
            similarity = unit_rows(bank_signature) @ (query / max(float(np.linalg.norm(query)), 1e-30))
            nearest_index = candidates[int(similarity.argmax())]; farthest_index = candidates[int(similarity.argmin())]
            rows_needed = [int(atlas[f"layer{source_layer}_row"][nearest_index]), int(atlas[f"layer{source_layer}_row"][farthest_index])]
            for reference_row in rows_needed:
                if reference_row not in reference_cache:
                    ids = torch.as_tensor(np.asarray(discovery[reference_row, :source["sequence_length"]], dtype=np.int64), device="cuda")[None]
                    with torch.no_grad(): reference_cache[reference_row] = forward_layers_native(model, ids, len(model.blocks) - 1)
            traces = {"target": target_trace, "nearest": reference_cache[rows_needed[0]], "farthest": reference_cache[rows_needed[1]]}
            delta = product_at(model, donor_trace, source_layer, row["donor_position"]) - product_at(model, target_trace, source_layer, position)
            exact = semantic_gradient(model, target_trace, source_layer, position, contrast)
            specs = specifications(source_layer, len(model.blocks), config["near_widths"]); predictions = hybrid_program(model, traces, source_layer, position, contrast, specs)
            predictions["exact"] = exact
            patches = {"base": torch.zeros_like(delta)}
            for name, prediction in predictions.items():
                support = (prediction * delta).topk(config["support_rank"]).indices; patches[name] = selected_delta(delta, support)
            with torch.no_grad(): logits = logits_for_deltas(model, source_layer, target_trace[source_layer], patches, position)
            base_score = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"layer": source_layer, "feature": row["feature"], "selection": row["selection"], "nearest_similarity": float(similarity.max()), "farthest_similarity": float(similarity.min()), "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in predictions.items()}, "finite_shifts": finite})
        print(f"state swap {site_index + 1}/{len(selected)} {row['feature']} {row['selection']}", flush=True)
    methods = list(records[0]["metrics"]); summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["source_layers"]}; cells = [summaries[str(layer)] for layer in config["source_layers"]]
    hypotheses = {
        "S0_exact_state_program_reconstructs_gradient": {"supported": all(cell["exact_all"]["gradient_cosine_median"] >= .999 and cell["exact_all"]["gradient_relative_error_median"] <= .02 for cell in cells)},
        "S1_nearest_reference_operator_is_reusable": {"supported": all(cell["nearest_all"]["gradient_cosine_median"] >= .50 for cell in cells)},
        "S2_one_operator_family_is_mostly_reusable": {"supported": all(max(cell["exact_mlp"]["gradient_cosine_median"], cell["exact_attention"]["gradient_cosine_median"]) >= .70 for cell in cells)},
        "S3_exact_near_eight_states_are_sufficient": {"supported": all(cell["exact_near_8"]["gradient_cosine_median"] >= .90 for cell in cells)},
        "S4_near_states_matter_more_than_late_states": {"supported": all(cell["exact_near_8"]["gradient_cosine_median"] - cell["exact_late_8"]["gradient_cosine_median"] >= .10 for cell in cells)},
        "S5_forward_similarity_selects_better_operator_state": {"supported": all(cell["nearest_all"]["gradient_cosine_median"] - cell["farthest_all"]["gradient_cosine_median"] >= .10 for cell in cells)},
        "S6_best_single_family_hybrid_is_finitely_directional": {"supported": all(max(cell["exact_mlp"]["finite_directional_fraction"], cell["exact_attention"]["finite_directional_fraction"]) >= .75 for cell in cells)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    display = ["nearest_all", "farthest_all", "exact_mlp", "exact_attention", "exact_mlp_only", "exact_attention_only", "residual_only", "exact_near_4", "exact_near_8", "exact_late_8", "exact_all"]
    lines = ["# Local-operator state-swap pilot across Layers 0–3", "", "The output objective and final RMS/unembedding adjoint remain fixed to the target example. Each suffix block's exact MLP and attention VJP is then evaluated either at the target state or at a same-position reference context. References are selected as the nearest or farthest atlas state under the complete current-position forward trajectory.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Local operator swaps](bilin18_layers0_3_local_operator_state_swap.png)", "", "| Layer | Nearest all | Farthest all | Exact MLP + wrong attn. | Exact attn. + wrong MLP | Exact MLP only | Exact attn. only | Residual only | Exact near 4 | Exact near 8 | Exact late 8 | Exact |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        cell = summaries[str(layer)]; lines.append("| " + str(layer) + " | " + " | ".join(f"{cell[name]['gradient_cosine_median']:.3f}" for name in display) + " |")
    lines += ["", "## Post-hoc poisoning controls", "", "The MLP-only, attention-only, and residual-only rows were added after inspecting the first swap result. They distinguish omission of an operator family from insertion of a wrong-context Jacobian and are exploratory, not locked hypotheses.", "", "## Guardrails", "", "This is a local-Jacobian swap, not a valid forward counterfactual trajectory: MLP and attention Jacobians from different contexts can be composed even though no natural input would realize the hybrid. That is intentional for necessity localization. Keeping the target final adjoint makes reuse easier, so failure of a swapped program is strong evidence that intermediate local states matter; success would not by itself provide a generative interpreter."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained"); x = np.arange(len(config["source_layers"]))
    for method in display[:-1]: axes[0].plot(x, [summaries[str(layer)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Which local states must remain exact?"); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8, ncol=2)
    for method in ("exact_mlp", "exact_attention", "exact_near_8", "exact_late_8"):
        axes[1].plot(x, [summaries[str(layer)][method]["finite_directional_fraction"] for layer in config["source_layers"]], marker="o", label=method)
    axes[1].set_xticks(x, config["source_layers"]); axes[1].set(xlabel="Source layer", ylabel="Finite directional fraction", title="Finite top-64 validation"); axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
