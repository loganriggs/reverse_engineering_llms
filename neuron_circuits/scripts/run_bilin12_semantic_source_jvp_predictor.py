#!/usr/bin/env python3
"""Predict counterfactual source programs with a base-point tensor-network JVP."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_semantic_source_program_transfer import jaccard, select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_semantic_source_jvp_predictor.json")
OUTPUT = TENSOR_ROOT / "bilin12_semantic_source_jvp_predictor.json"
REPORT = TENSOR_ROOT / "BILIN12_SEMANTIC_SOURCE_JVP_PREDICTOR.md"
FIGURE = TENSOR_ROOT / "bilin12_semantic_source_jvp_predictor.png"


def _rotary_constants(block, length: int, device: torch.device, dtype: torch.dtype):
    inv_freq = block.attn.rotary.inv_freq.to(device=device, dtype=torch.float32)
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    # The checkpoint's native implementation stores these constants in bf16.
    return frequencies.cos().to(torch.bfloat16).to(dtype)[None, :, None], frequencies.sin().to(torch.bfloat16).to(dtype)[None, :, None]


def differentiable_attention(model, block, normalized, value_bus):
    batch, length, dimension = normalized.shape
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    query = block.attn.c_q(normalized).view(batch, length, heads, head_dimension)
    key = block.attn.c_k(normalized).view(batch, length, heads, head_dimension)
    current_value = block.attn.c_v(normalized).view(batch, length, heads, head_dimension)
    first_value = current_value if value_bus is None else value_bus
    cosine, sine = _rotary_constants(block, length, normalized.device, normalized.dtype)
    query = model.tt.apply_rotary_emb(F.rms_norm(query, (head_dimension,)), cosine, sine)
    key = model.tt.apply_rotary_emb(F.rms_norm(key, (head_dimension,)), cosine, sine)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dimension
    causal = torch.tril(torch.ones(length, length, device=normalized.device, dtype=torch.bool))
    unnormalized_pattern = scores.square().masked_fill(~causal, 0.0)
    pattern_mass = unnormalized_pattern.sum(-1, keepdim=True)
    pattern = unnormalized_pattern / pattern_mass.clamp_min(1e-9)
    mixed_value = (1 - block.attn.lamb) * current_value + block.attn.lamb * first_value
    attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value)
    write = block.attn.c_proj(attended.transpose(1, 2).contiguous().view(batch, length, dimension))
    return write, first_value, pattern, mixed_value, pattern_mass


def projected_source_writes(block, pattern, mixed_value, destination: int):
    """Exact bias-free source contributions for one destination."""
    per_head = pattern[:, :, destination, :, None] * mixed_value.permute(0, 2, 1, 3)
    pieces = []
    for head in range(block.attn.n_head):
        start = head * block.attn.head_dim
        stop = (head + 1) * block.attn.head_dim
        pieces.append(per_head[:, head] @ block.attn.c_proj.weight[:, start:stop].T)
    return torch.stack(pieces).sum(0)


def differentiable_source_trace(model, x0, layers, destination: int):
    """Trace from normalized embeddings and return source tensors at layers."""
    requested = set(layers)
    x = x0
    value_bus = None
    outputs = []
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (mixed.shape[-1],))
        write, value_bus, pattern, mixed_value, _ = differentiable_attention(model, block, normalized, value_bus)
        if layer in requested:
            outputs.append(projected_source_writes(block, pattern, mixed_value, destination))
        pre = mixed + write
        state = F.rms_norm(pre, (pre.shape[-1],))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
    return tuple(outputs)


def differentiable_source_trace_with_masses(model, x0, layers, destination: int):
    """Return requested source tensors and their raw squared-QK normalizers."""
    requested = set(layers)
    x = x0
    value_bus = None
    sources = []
    masses = []
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (mixed.shape[-1],))
        write, value_bus, pattern, mixed_value, pattern_mass = differentiable_attention(model, block, normalized, value_bus)
        if layer in requested:
            sources.append(projected_source_writes(block, pattern, mixed_value, destination))
            masses.append(pattern_mass[:, :, destination, 0])
        pre = mixed + write
        state = F.rms_norm(pre, (pre.shape[-1],))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
    return tuple(sources), tuple(masses)


def source_jvp(model, base_ids, changed_ids, layers, destination: int):
    dimension = model.config.n_embd
    base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,))
    changed_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,))
    tangent = changed_x0 - base_x0

    def function(value):
        return differentiable_source_trace(model, value, layers, destination)

    primal, linear = torch.func.jvp(function, (base_x0,), (tangent,))
    return base_x0, tangent, primal, linear


def cosine(left, right):
    left = left.flatten(1)
    right = right.flatten(1)
    return (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-30)


def metric(base_logits, target_delta, predicted_delta):
    target, _, _, error = fisher_terms(base_logits, target_delta, predicted_delta)
    return {
        "target_energy": float(target.sum()),
        "error": float(error.sum()),
        "fisher_fidelity": float(1 - error.sum() / target.sum().clamp_min(1e-30)),
    }


def analyze_example(model, base_ids, changed_ids, config):
    layers = config["layers"]
    destination = config["prediction_position"]
    with torch.no_grad():
        base_rows, _, _ = trace(model, base_ids)
        changed_rows, _, _ = trace(model, changed_ids)
    with torch.enable_grad():
        base_x0, input_tangent, manual_base, linear = source_jvp(model, base_ids, changed_ids, layers, destination)

    results = {}
    for layer_index, layer in enumerate(layers):
        block = model.blocks[layer]
        base = base_rows[layer]
        changed = changed_rows[layer]
        exact_base = source_write_contributions(block, base["details"], destination).float()
        exact_changed = source_write_contributions(block, changed["details"], destination).float()
        exact_delta = exact_changed - exact_base
        predicted_delta = linear[layer_index].float()
        manual_error = max(
            float((manual_base[layer_index].float() - exact_base).norm() / exact_base.norm().clamp_min(1e-30)),
            float(
                (differentiable_source_trace(model, F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,)), [layer], destination)[0].float() - exact_changed).norm()
                / exact_changed.norm().clamp_min(1e-30)
            ),
        )

        full_write = base["details"]["write"].clone()
        full_write[:, destination] = changed["details"]["write"][:, destination]
        base_logits, target_logits = logits_from_attention_writes(
            model, layer, base, torch.stack((base["details"]["write"], full_write))
        )
        target_delta = target_logits - base_logits

        base_mass = base["details"]["pattern"][:, :, destination].sum(1)
        tangent_direction = input_tangent[:, config["edit_position"]]
        input_similarity = torch.einsum("bsd,bd->bs", base_x0, tangent_direction).abs()
        orders = {
            "exact_state_norm": exact_delta.norm(dim=-1).argsort(-1, descending=True),
            "jvp_state_norm": predicted_delta.norm(dim=-1).argsort(-1, descending=True),
            "base_source_norm": exact_base.norm(dim=-1).argsort(-1, descending=True),
            "base_attention_mass": base_mass.argsort(-1, descending=True),
            "input_tangent_similarity": input_similarity.argsort(-1, descending=True),
            "recency": torch.arange(config["sequence_length"] - 1, -1, -1, device=base_ids.device)[None],
        }

        worlds = []
        labels = []
        for method in config["selection_methods"]:
            for budget in config["budgets"]:
                indices = orders[method][:, :budget]
                mask = torch.zeros(1, config["sequence_length"], device=base_ids.device)
                mask.scatter_(1, indices, 1)
                approximation = (mask[:, :, None] * exact_delta).sum(1)
                write = base["details"]["write"].clone()
                write[:, destination] += approximation.to(write.dtype)
                worlds.append(write)
                labels.append(f"select_exact/{method}/{budget}")

                if method == "jvp_state_norm":
                    predicted = (mask[:, :, None] * predicted_delta).sum(1)
                    predicted_write = base["details"]["write"].clone()
                    predicted_write[:, destination] += predicted.to(predicted_write.dtype)
                    worlds.append(predicted_write)
                    labels.append(f"insert_jvp/{budget}")

        edited_write = base["details"]["write"].clone()
        edited_write[:, destination] += exact_delta[:, config["edit_position"]].to(edited_write.dtype)
        worlds.append(edited_write)
        labels.append("edited_source")

        logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
        metrics = {label: metric(base_logits, target_delta, logits[index] - base_logits) for index, label in enumerate(labels)}
        exact_top16 = orders["exact_state_norm"][0, :16].cpu().tolist()
        jvp_top16 = orders["jvp_state_norm"][0, :16].cpu().tolist()
        results[str(layer)] = {
            "manual_source_relative_error": manual_error,
            "jvp_exact_cosine": float(cosine(predicted_delta, exact_delta)[0]),
            "jvp_exact_relative_error": float((predicted_delta - exact_delta).norm() / exact_delta.norm().clamp_min(1e-30)),
            "top16_jaccard": jaccard(exact_top16, jvp_top16),
            "exact_top16": exact_top16,
            "jvp_top16": jvp_top16,
            "metrics": metrics,
        }
    return results


def aggregate(records, config, selector=lambda row: True):
    totals = defaultdict(lambda: {"target": 0.0, "error": 0.0, "count": 0, "values": []})
    for record in records:
        if not selector(record):
            continue
        for layer in record["layers"].values():
            for label, row in layer["metrics"].items():
                total = totals[label]
                total["target"] += row["target_energy"]
                total["error"] += row["error"]
                total["count"] += 1
                total["values"].append(row["fisher_fidelity"])
    return {
        label: {
            "examples": row["count"],
            "fisher_fidelity": 1 - row["error"] / max(row["target"], 1e-30),
            "mean_individual_fidelity": float(np.mean(row["values"])),
            "q05_individual_fidelity": float(np.quantile(row["values"], 0.05)),
            "minimum_individual_fidelity": float(np.min(row["values"])),
        }
        for label, row in totals.items()
    }


def main():
    config = json.loads(CONFIG.read_text())
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    records = []
    for corpus_index, (corpus, spec) in enumerate(config["evaluation"].items()):
        source = np.load(spec["path"], mmap_mode="r")
        rows = source[spec["start"] : spec["start"] + spec["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        for family, pairs in semantic["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows,
                    source_token,
                    config["edit_position"],
                    config["prediction_position"],
                    config["contexts_per_pair"],
                    rng,
                )
                for row_index, start in windows:
                    base = np.asarray(rows[row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = base.copy()
                    changed[config["edit_position"]] = target_token
                    layers = analyze_example(
                        model,
                        torch.as_tensor(base, device="cuda")[None],
                        torch.as_tensor(changed, device="cuda")[None],
                        config,
                    )
                    records.append(
                        {
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "source_token": tokenizer.decode([source_token]),
                            "target_token": tokenizer.decode([target_token]),
                            "row": int(row_index),
                            "start": int(start),
                            "layers": layers,
                        }
                    )
                    print(corpus, family, tokenizer.decode([source_token]), "->", tokenizer.decode([target_token]), flush=True)

    overall = aggregate(records, config)
    by_layer = {str(layer): aggregate(records, config, lambda record, layer=layer: True) for layer in config["layers"]}
    # The layer selector is applied below because aggregate traverses all layers.
    def layer_summary(layer):
        projected = []
        for record in records:
            projected.append({**record, "layers": {str(layer): record["layers"][str(layer)]}})
        return aggregate(projected, config)

    by_layer = {str(layer): layer_summary(layer) for layer in config["layers"]}
    by_corpus = {corpus: aggregate(records, config, lambda row, corpus=corpus: row["corpus"] == corpus) for corpus in config["evaluation"]}
    by_family = {family: aggregate(records, config, lambda row, family=family: row["family"] == family) for family in semantic["families"]}

    all_layers = [layer for record in records for layer in record["layers"].values()]
    cosines = np.asarray([layer["jvp_exact_cosine"] for layer in all_layers])
    overlaps = np.asarray([layer["top16_jaccard"] for layer in all_layers])
    manual_errors = np.asarray([layer["manual_source_relative_error"] for layer in all_layers])
    jvp32 = overall["select_exact/jvp_state_norm/32"]
    exact32 = overall["select_exact/exact_state_norm/32"]
    jvp16 = overall["select_exact/jvp_state_norm/16"]
    static16 = [
        overall[f"select_exact/{method}/16"]["fisher_fidelity"]
        for method in ("base_source_norm", "base_attention_mass", "input_tangent_similarity", "recency")
    ]
    subgroup32 = [
        summary["select_exact/jvp_state_norm/32"]["fisher_fidelity"]
        for collection in (by_layer, by_corpus, by_family)
        for summary in collection.values()
    ]
    hypotheses = {
        "J1_manual_differentiable_trace_matches_native_sources": {"supported": bool(manual_errors.max() <= 5e-5), "maximum_relative_error": float(manual_errors.max())},
        "J2_jvp_source_delta_has_high_directional_agreement": {"supported": bool(np.median(cosines) >= 0.80), "median_cosine": float(np.median(cosines)), "q05_cosine": float(np.quantile(cosines, 0.05))},
        "J3_jvp_ranking_selects_a_high_fidelity_program": {"supported": bool(jvp32["fisher_fidelity"] >= 0.90 and jvp32["q05_individual_fidelity"] >= 0.75), "aggregate": jvp32["fisher_fidelity"], "q05": jvp32["q05_individual_fidelity"]},
        "J4_jvp_selection_is_close_to_counterfactual_selection": {"supported": bool(exact32["fisher_fidelity"] - jvp32["fisher_fidelity"] <= 0.05), "gap": exact32["fisher_fidelity"] - jvp32["fisher_fidelity"]},
        "J5_jvp_prediction_reconstructs_behavior_not_only_ranking": {"supported": bool(overall["insert_jvp/64"]["fisher_fidelity"] >= 0.75), "fidelity": overall["insert_jvp/64"]["fisher_fidelity"]},
        "J6_jvp_beats_static_base_shortcuts": {"supported": bool(jvp16["fisher_fidelity"] - max(static16) >= 0.10), "advantage": jvp16["fisher_fidelity"] - max(static16), "best_static": max(static16)},
        "J7_jvp_selection_transfers_across_layers_corpora_and_families": {"supported": bool(min(subgroup32) >= 0.75), "minimum": min(subgroup32)},
        "J8_jvp_and_exact_programs_overlap": {"supported": bool(np.median(overlaps) >= 0.40), "median_jaccard": float(np.median(overlaps))},
        "J9_input_similarity_alone_is_not_the_explanation": {"supported": bool(jvp16["fisher_fidelity"] - overall["select_exact/input_tangent_similarity/16"]["fisher_fidelity"] >= 0.15), "advantage": jvp16["fisher_fidelity"] - overall["select_exact/input_tangent_similarity/16"]["fisher_fidelity"]},
        "J10_predictive_success_is_not_confined_to_the_edited_position": {"supported": bool(overall["edited_source"]["fisher_fidelity"] < 0.50 and jvp16["fisher_fidelity"] >= 0.80), "edited_source": overall["edited_source"]["fisher_fidelity"], "jvp16": jvp16["fisher_fidelity"]},
    }

    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "overall": overall,
        "by_layer": by_layer,
        "by_corpus": by_corpus,
        "by_family": by_family,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    budgets = config["budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for method in config["selection_methods"]:
        axes[0].plot(budgets, [overall[f"select_exact/{method}/{budget}"]["fisher_fidelity"] for budget in budgets], marker="o", label=method)
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Program selection using exact retained edges")
    axes[0].set_xlabel("source budget")
    axes[0].set_ylabel("Fisher fidelity")
    axes[1].plot(budgets, [overall[f"select_exact/jvp_state_norm/{budget}"]["fisher_fidelity"] for budget in budgets], marker="o", label="exact deltas, JVP selection")
    axes[1].plot(budgets, [overall[f"insert_jvp/{budget}"]["fisher_fidelity"] for budget in budgets], marker="o", label="JVP deltas")
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Selection versus full linear reconstruction")
    axes[1].set_xlabel("source budget")
    axes[1].set_ylabel("Fisher fidelity")
    axes[2].hist(cosines, bins=16, alpha=0.7, label="JVP/exact cosine")
    axes[2].hist(overlaps, bins=16, alpha=0.5, label="top-16 Jaccard")
    axes[2].set_title("Individual-example agreement")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Base-point JVP prediction of semantic source programs",
        "",
        "The normalized embedding difference between each lexical pair is propagated once through a forward-mode Jacobian-vector product of the tensor network. This predicts every layer's final-destination source contribution change from the base point and input tangent, without running the finite counterfactual state. Selection fidelity uses exact finite deltas only after the JVP has selected positions; reconstruction fidelity inserts the JVP-predicted deltas themselves.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic source JVP predictor](bilin12_semantic_source_jvp_predictor.png)",
        "",
        "## Aggregate program frontier",
        "",
        "| Budget | Exact-state selector | JVP selector | JVP reconstruction | Base source | Base attention | Input similarity | Recency |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        lines.append(
            f"| {budget} | {overall[f'select_exact/exact_state_norm/{budget}']['fisher_fidelity']:.3f} | {overall[f'select_exact/jvp_state_norm/{budget}']['fisher_fidelity']:.3f} | {overall[f'insert_jvp/{budget}']['fisher_fidelity']:.3f} | {overall[f'select_exact/base_source_norm/{budget}']['fisher_fidelity']:.3f} | {overall[f'select_exact/base_attention_mass/{budget}']['fisher_fidelity']:.3f} | {overall[f'select_exact/input_tangent_similarity/{budget}']['fisher_fidelity']:.3f} | {overall[f'select_exact/recency/{budget}']['fisher_fidelity']:.3f} |"
        )
    lines += [
        "",
        f"Median JVP/exact source-tensor cosine: **{np.median(cosines):.3f}**. Median exact/JVP top-16 Jaccard: **{np.median(overlaps):.3f}**. Maximum manual-source recomposition relative error: **{manual_errors.max():.2e}**.",
        "",
        "The JVP predictor knows the proposed token replacement through its normalized embedding tangent, but it does not use the finite changed hidden states, target logits, unembedding, or downstream Fisher direction. It is therefore a local mechanistic predictor, not a context-free semantic circuit and not a zero-information forecast.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
