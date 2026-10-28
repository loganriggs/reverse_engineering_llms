#!/usr/bin/env python3
"""Compose frozen operator-tensor modes inside exact semantic suffix programs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_dynamic_mlp_jacobian_spectrum import exact_incoming_adjoints
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_operator_vocabulary_causal_composition.json")
OUTPUT = TENSOR_ROOT / "bilin18_operator_vocabulary_causal_composition.json"
REPORT = TENSOR_ROOT / "BILIN18_OPERATOR_VOCABULARY_CAUSAL_COMPOSITION.md"
FIGURE = TENSOR_ROOT / "bilin18_operator_vocabulary_causal_composition.png"


def reconstruct_state(state, basis, rank, family):
    mean = basis["mean"].to(state.device)
    vectors = basis["state_vectors"].to(state.device)
    if family == "pca":
        selected = vectors[:, :rank]
        return mean + ((state - mean) @ selected) @ selected.T
    rms = basis["state_rms"].to(state.device)
    modes = basis[f"operator_modes_{family}"].to(state.device)[:, :rank]
    whitened = ((state - mean) @ vectors) / rms
    recovered = (whitened @ modes) @ modes.T
    return mean + (recovered * rms) @ vectors.T


def approximate_mlp_vjp(block, pre, incoming, normalized_state):
    """Exact weight formula, replacing only the state that conditions J_MLP."""
    x = pre.float(); dimension = x.numel(); rms = x.square().mean().sqrt()
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down_weight = block.mlp.Down.weight.float()
    left = left_weight @ normalized_state
    right = right_weight @ normalized_state
    channel = incoming @ down_weight
    core = (channel * right) @ left_weight + (channel * left) @ right_weight
    return core / rms - x * (core @ x) / (dimension * rms.pow(3))


def program(model, trace, position, contrast, source_layer, bases, config):
    specs = [("exact", None, None)]
    for rank in config["ranks"]:
        specs += [(f"operator_first_{rank}", "first", rank), (f"operator_second_{rank}", "second", rank), (f"pca_{rank}", "pca", rank)]
    current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(len(specs), 1, 1, 1)
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        responses = mlp_vjp(block, trace[layer]["pre"].float(), current)
        if layer in config["approximated_blocks"] and str(layer) in bases:
            pre = trace[layer]["pre"][0, position].float()
            state = F.rms_norm(pre, (pre.numel(),))
            for row, (_, family, rank) in enumerate(specs):
                if family is None: continue
                reconstructed = reconstruct_state(state, bases[str(layer)], rank, family)
                responses[row, 0, position] = approximate_mlp_vjp(
                    block, pre, current[row, 0, position], reconstructed,
                )
        updated = current + responses
        previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, position] @ down for row, (name, _, _) in enumerate(specs)}


def select_sites(causal, old, sites_per_feature):
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]
        extreme = [row for row in rows if row["selection"] == "extreme"]
        uniform = [row for row in rows if row["selection"] == "uniform"]
        first = sites_per_feature // 2
        selected.extend(extreme[:first]); selected.extend(uniform[:sites_per_feature - first])
    return selected


def summarize_program(records, layer, method):
    rows = [row for row in records if row["source_layer"] == layer]
    values = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])),
        "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)),
        "contribution_cosine_median": float(np.median([value["contribution_cosine"] for value in values])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    bases = torch.load(config["bases"], map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    sites = select_sites(causal, old, config["sites_per_feature"])
    records, local = [], []
    maximum_error = 0.0
    for site_index, row in enumerate(sites):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]; position = row["recipient_position"]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        incoming = exact_incoming_adjoints(model, trace, position, contrast, first_block=min(config["approximated_blocks"]))
        for block_index in config["approximated_blocks"]:
            pre = trace[block_index]["pre"][0, position].float(); state = F.rms_norm(pre, (pre.numel(),))
            native = mlp_vjp(model.blocks[block_index], trace[block_index]["pre"].float(), incoming[block_index])[0, 0, position]
            explicit = approximate_mlp_vjp(model.blocks[block_index], pre, incoming[block_index][0, 0, position], state)
            maximum_error = max(maximum_error, float((native - explicit).norm() / native.norm().clamp_min(1e-30)))
            methods = {}
            for rank in config["ranks"]:
                for family in ("first", "second", "pca"):
                    reconstructed = reconstruct_state(state, bases[str(block_index)], rank, family)
                    predicted = approximate_mlp_vjp(model.blocks[block_index], pre, incoming[block_index][0, 0, position], reconstructed)
                    methods[f"{family}_{rank}"] = float(F.cosine_similarity(predicted, native, dim=0))
            local.append({"site": site_index, "feature": row["feature"], "block": block_index, "cosines": methods})
        for source_layer in config["source_layers"]:
            exact = semantic_gradient(model, trace, source_layer, position, contrast)
            predictions = program(model, trace, position, contrast, source_layer, bases, config)
            predictions["direct_exact"] = exact
            basis_agreement = {
                str(rank): float(F.cosine_similarity(
                    predictions[f"operator_first_{rank}"],
                    predictions[f"operator_second_{rank}"], dim=0,
                ))
                for rank in config["ranks"]
            }
            delta = product_at(model, donor_trace, source_layer, row["donor_position"]) - product_at(model, trace, source_layer, position)
            patches = {"base": torch.zeros_like(delta)}
            for name, prediction in predictions.items():
                support = (prediction * delta).topk(config["support_rank"]).indices
                patches[name] = selected_delta(delta, support)
            logits = logits_for_deltas(model, source_layer, trace[source_layer], patches, position)
            base_score = class_score(logits["base"], contrast)
            finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"site": site_index, "feature": row["feature"], "source_layer": source_layer, "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in predictions.items()}, "basis_agreement": basis_agreement, "finite_shifts": finite})
        print(f"operator causal {site_index + 1}/{len(sites)} {row['feature']}", flush=True)

    local_summary = {}
    for block in config["approximated_blocks"]:
        rows = [row for row in local if row["block"] == block]
        local_summary[str(block)] = {
            name: float(np.median([row["cosines"][name] for row in rows]))
            for name in rows[0]["cosines"]
        }
    program_summary = {}
    methods = list(records[0]["metrics"])
    for layer in config["source_layers"]:
        program_summary[str(layer)] = {method: summarize_program(records, layer, method) for method in methods}
        selected = [row for row in records if row["source_layer"] == layer]
        program_summary[str(layer)]["rank64_basis_agreement_median"] = float(np.median([row["basis_agreement"]["64"] for row in selected]))
    hypotheses = {
        "J0_weight_formula_matches_native_vjp": {"supported": maximum_error <= 1e-5},
        "J1_rank256_reconstructs_task_local_responses": {"supported": all(local_summary[str(block)]["first_256"] >= .90 for block in config["approximated_blocks"])},
        "J2_rank256_composes_through_suffix": {"supported": all(program_summary[str(layer)]["operator_first_256"]["gradient_cosine_median"] >= .90 for layer in config["source_layers"])},
        "J3_operator_rank64_beats_pca_causally": {"supported": all(program_summary[str(layer)]["operator_first_64"]["gradient_cosine_median"] - program_summary[str(layer)]["pca_64"]["gradient_cosine_median"] >= .02 for layer in config["source_layers"])},
        "J4_operator_rank64_is_finitely_directional": {"supported": all(program_summary[str(layer)]["operator_first_64"]["finite_directional_fraction"] >= .90 for layer in config["source_layers"])},
        "J5_independent_operator_bases_agree_causally": {"supported": all(program_summary[str(layer)]["rank64_basis_agreement_median"] >= .95 for layer in config["source_layers"])},
        "J6_operator_modes_beat_pca_on_task_responses": {"supported": sum(local_summary[str(block)]["first_64"] > local_summary[str(block)]["pca_64"] for block in config["approximated_blocks"]) >= 4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_formula_error": maximum_error, "hypotheses": hypotheses, "local_summary": local_summary, "program_summary": program_summary, "local_records": local, "records": records}, indent=2, allow_nan=False))
    lines = ["# Causal composition of the early operator vocabulary", "", "The converged, label-free operator-state bases are frozen. At Blocks 1--5, the native state conditioning each bilinear MLP Jacobian is replaced by its operator-basis or state-PCA reconstruction; the actual RMSNorm pullback, all other positions, every attention derivative, and the remaining suffix stay exact. The resulting source-product gradients select top-64 native channels prospectively and are validated by finite suffix recomputation.", "", f"Maximum explicit weight-formula versus native VJP error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Causal operator composition](bilin18_operator_vocabulary_causal_composition.png)", "", "| Block | Op64 A/B | PCA64 | Op256 | PCA256 |", "|---:|---:|---:|---:|---:|"]
    for block in config["approximated_blocks"]:
        row = local_summary[str(block)]; lines.append(f"| {block} | {row['first_64']:.3f}/{row['second_64']:.3f} | {row['pca_64']:.3f} | {row['first_256']:.3f} | {row['pca_256']:.3f} |")
    lines += ["", "| Source layer | Op64 A/B gradient cosine | A↔B cosine | PCA64 | Op256 | PCA256 | Op64 finite direction |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        row = program_summary[str(layer)]; lines.append(f"| {layer} | {row['operator_first_64']['gradient_cosine_median']:.3f}/{row['operator_second_64']['gradient_cosine_median']:.3f} | {row['rank64_basis_agreement_median']:.3f} | {row['pca_64']['gradient_cosine_median']:.3f} | {row['operator_first_256']['gradient_cosine_median']:.3f} | {row['pca_256']['gradient_cosine_median']:.3f} | {row['operator_first_64']['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrails", "", "Only Blocks 1--5 are compressed, so high composed fidelity does not claim that the entire suffix is rank 64 or 256. Conversely, a local Frobenius reconstruction can fail on a particular semantic adjoint even when its average operator R2 is good. Finite validation tests the selected source-channel intervention, not replacement of the model's forward MLP computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    blocks = config["approximated_blocks"]
    for name in ("first_64", "second_64", "pca_64", "first_256", "pca_256"):
        axes[0].plot(blocks, [local_summary[str(block)][name] for block in blocks], marker="o", label=name)
    axes[0].set(xlabel="Approximated block", ylabel="Median local response cosine", title="Task-adjoint local reconstruction"); axes[0].grid(alpha=.2); axes[0].legend()
    layers = config["source_layers"]
    for name in ("operator_first_64", "operator_second_64", "pca_64", "operator_first_256", "pca_256"):
        axes[1].plot(layers, [program_summary[str(layer)][name]["gradient_cosine_median"] for layer in layers], marker="o", label=name)
    axes[1].set(xlabel="Source layer", ylabel="Median source-gradient cosine", title="Composition through exact suffix"); axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"maximum_formula_error": maximum_error, "hypotheses": hypotheses, "local_summary": local_summary, "program_summary": program_summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
