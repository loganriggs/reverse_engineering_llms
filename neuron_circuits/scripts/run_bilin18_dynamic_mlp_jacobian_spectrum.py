#!/usr/bin/env python3
"""Spectral and causal audit of state-conditioned bilinear MLP Jacobians."""

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
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_dynamic_mlp_jacobian_spectrum.json")
OUTPUT = TENSOR_ROOT / "bilin18_dynamic_mlp_jacobian_spectrum.json"
REPORT = TENSOR_ROOT / "BILIN18_DYNAMIC_MLP_JACOBIAN_SPECTRUM.md"
FIGURE = TENSOR_ROOT / "bilin18_dynamic_mlp_jacobian_spectrum.png"


def analytic_mlp_jacobian(block, pre_vector, coefficient_permutation=None):
    """Forward Jacobian d(MLP write)/d(pre), including the RMSNorm pullback."""
    x = pre_vector.float(); dimension = x.numel(); rms = x.square().mean().sqrt()
    state = x / rms
    left = F.linear(state, block.mlp.Left.weight.float())
    right = F.linear(state, block.mlp.Right.weight.float())
    if coefficient_permutation is not None:
        left = left[coefficient_permutation]; right = right[coefficient_permutation]
    channel_derivative = right[:, None] * block.mlp.Left.weight.float() + left[:, None] * block.mlp.Right.weight.float()
    core = block.mlp.Down.weight.float() @ channel_derivative
    core_x = core @ x
    return core / rms - torch.outer(core_x, x) / (dimension * rms.pow(3))


def cosine(left, right):
    return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))


def k90(values):
    values = values.square(); total = values.sum()
    if float(total) <= 0: return 0
    return int(torch.searchsorted(values.cumsum(0), .9 * total).item()) + 1


def exact_incoming_adjoints(model, trace, position, contrast, first_block=1):
    # The shared VJP helpers support a batch of candidate backward programs in
    # their leading axis.  The exact program is a singleton candidate.
    current = final_adjoint(model, trace[-1]["output"], position, contrast).unsqueeze(0)
    incoming = {}
    for layer in range(len(model.blocks) - 1, first_block - 1, -1):
        incoming[layer] = current.detach().clone()
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
    return incoming


def truncated_response(vector, decomposition, rank, selection="spectral"):
    u, singular, vh = decomposition
    coefficients = (vector @ u) * singular
    if selection == "task":
        indices = coefficients.abs().topk(rank).indices
        return coefficients[indices] @ vh[indices]
    return coefficients[:rank] @ vh[:rank]


def truncated_suffix_program(model, trace, position, contrast, decompositions, blocks, ranks, source_layer=0):
    blocks = [block for block in blocks if block > source_layer]
    specs = [("exact", None, None, "spectral")]
    for rank in ranks:
        specs.append((f"all_rank_{rank}", set(blocks), rank, "spectral"))
        specs.append((f"all_task_rank_{rank}", set(blocks), rank, "task"))
        for block in blocks:
            specs.append((f"block_{block}_rank_{rank}", {block}, rank, "spectral"))
    current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(len(specs), 1, 1, 1)
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        responses = mlp_vjp(block, trace[layer]["pre"].float(), current)
        if layer in blocks:
            decomposition = decompositions[layer]
            for row, (_, selected_blocks, rank, selection) in enumerate(specs):
                if selected_blocks is not None and layer in selected_blocks:
                    responses[row, 0, position] = truncated_response(current[row, 0, position], decomposition, rank, selection)
        updated = current + responses
        previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, position] @ down for row, (name, _, _, _) in enumerate(specs)}


def principal_overlap(first, second):
    rank = first.shape[1]
    return float((first.T @ second).square().sum() / rank)


def summarize(values):
    return {"median": float(np.median(values)), "q10": float(np.quantile(values, .1)), "q90": float(np.quantile(values, .9))}


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    selected = []
    for feature_name in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature_name]
        selected.append(next(row for row in rows if row["selection"] == "extreme"))
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    local_records = []; causal_records = []; mode_store = {str(block): [] for block in config["blocks"]}; maximum_error = 0.0
    for site_index, row in enumerate(selected):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]; position = row["recipient_position"]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config.get("exploratory_source_layers", [config["source_layer"]]))); trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        incoming = exact_incoming_adjoints(model, trace, position, contrast)
        decompositions = {}
        for layer in config["blocks"]:
            block = model.blocks[layer]; pre = trace[layer]["pre"][0, position].float(); matrix = analytic_mlp_jacobian(block, pre)
            u, singular, vh = torch.linalg.svd(matrix, full_matrices=False); decompositions[layer] = (u, singular, vh)
            vector = incoming[layer][0, 0, position]; exact_response = mlp_vjp(block, trace[layer]["pre"].float(), incoming[layer])[0, 0, position]
            analytic_response = vector @ matrix; error = float((analytic_response - exact_response).norm() / exact_response.norm().clamp_min(1e-30)); maximum_error = max(maximum_error, error)
            coefficients = (vector @ u) * singular
            task_order = coefficients.abs().argsort(descending=True)
            rank_metrics = {}
            for rank in config["ranks"]:
                approximation = truncated_response(vector, decompositions[layer], rank)
                task_approximation = truncated_response(vector, decompositions[layer], rank, "task")
                rank_metrics[str(rank)] = {"response_cosine": cosine(approximation, exact_response), "spectral_energy": float(singular[:rank].square().sum() / singular.square().sum()), "task_energy": float(coefficients[:rank].square().sum() / coefficients.square().sum().clamp_min(1e-30)), "task_selected_response_cosine": cosine(task_approximation, exact_response), "task_selected_energy": float(coefficients[task_order[:rank]].square().sum() / coefficients.square().sum().clamp_min(1e-30))}
            null_leads = []
            for _ in range(config["null_replicates"]):
                permutation = torch.randperm(block.mlp.Left.weight.shape[0], generator=generator, device="cuda")
                null_matrix = analytic_mlp_jacobian(block, pre, permutation)
                null_leads.append(float(torch.linalg.svdvals(null_matrix)[0]))
            rank = config["subspace_rank"]; mode_store[str(layer)].append({"u": u[:, :rank].cpu(), "v": vh[:rank].T.cpu()})
            local_records.append({"site": site_index, "feature": row["feature"], "block": layer, "analytic_relative_error": error, "leading_singular": float(singular[0]), "null_leading": null_leads, "leading_to_null_ratio": float(singular[0]) / max(null_leads), "spectral_k90": k90(singular), "task_prefix_k90": k90(coefficients.abs()), "task_optimal_k90": k90(coefficients[task_order].abs()), "effective_rank": float(torch.exp(-(singular.square() / singular.square().sum() * (singular.square() / singular.square().sum()).clamp_min(1e-30).log()).sum())), "rank_metrics": rank_metrics})
        for source_layer in config.get("exploratory_source_layers", [config["source_layer"]]):
            exact = semantic_gradient(model, trace, source_layer, position, contrast); predictions = truncated_suffix_program(model, trace, position, contrast, decompositions, config["blocks"], config["causal_ranks"], source_layer); predictions["direct_exact"] = exact
            delta = product_at(model, donor_trace, source_layer, row["donor_position"]) - product_at(model, trace, source_layer, position)
            patches = {"base": torch.zeros_like(delta)}
            for name, prediction in predictions.items():
                support = (prediction * delta).topk(config["support_rank"]).indices; patches[name] = selected_delta(delta, support)
            with torch.no_grad(): logits = logits_for_deltas(model, source_layer, trace[source_layer], patches, position)
            base_score = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            causal_records.append({"site": site_index, "feature": row["feature"], "source_layer": source_layer, "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in predictions.items()}, "finite_shifts": finite})
        print(f"dynamic MLP spectrum {site_index + 1}/{len(selected)} {row['feature']}", flush=True)
    block_summaries = {}
    random_expectation = config["subspace_rank"] / model.config.n_embd
    for block in config["blocks"]:
        rows = [row for row in local_records if row["block"] == block]; modes = mode_store[str(block)]; u_overlap = []; v_overlap = []
        for first in range(len(modes)):
            for second in range(first + 1, len(modes)):
                u_overlap.append(principal_overlap(modes[first]["u"], modes[second]["u"])); v_overlap.append(principal_overlap(modes[first]["v"], modes[second]["v"]))
        block_summaries[str(block)] = {"analytic_error_max": max(row["analytic_relative_error"] for row in rows), "leading_to_null_ratio": summarize([row["leading_to_null_ratio"] for row in rows]), "spectral_k90": summarize([row["spectral_k90"] for row in rows]), "task_prefix_k90": summarize([row["task_prefix_k90"] for row in rows]), "task_optimal_k90": summarize([row["task_optimal_k90"] for row in rows]), "effective_rank": summarize([row["effective_rank"] for row in rows]), "u_subspace_overlap": summarize(u_overlap), "v_subspace_overlap": summarize(v_overlap), "random_subspace_overlap": random_expectation, "ranks": {str(rank): {"response_cosine": summarize([row["rank_metrics"][str(rank)]["response_cosine"] for row in rows]), "spectral_energy": summarize([row["rank_metrics"][str(rank)]["spectral_energy"] for row in rows]), "task_energy": summarize([row["rank_metrics"][str(rank)]["task_energy"] for row in rows]), "task_selected_response_cosine": summarize([row["rank_metrics"][str(rank)]["task_selected_response_cosine"] for row in rows]), "task_selected_energy": summarize([row["rank_metrics"][str(rank)]["task_selected_energy"] for row in rows])} for rank in config["ranks"]}}
    causal_summaries = {}
    for source_layer in config.get("exploratory_source_layers", [config["source_layer"]]):
        source_rows = [row for row in causal_records if row["source_layer"] == source_layer]; causal_summaries[str(source_layer)] = {}
        for method in source_rows[0]["metrics"]:
            rows = [row["metrics"][method] for row in source_rows]; shifts = np.asarray([row["finite_shifts"][method] for row in source_rows]); causal_summaries[str(source_layer)][method] = {"gradient_cosine_median": float(np.median([row["gradient_cosine"] for row in rows])), "gradient_cosine_q10": float(np.quantile([row["gradient_cosine"] for row in rows], .1)), "contribution_cosine_median": float(np.median([row["contribution_cosine"] for row in rows])), "true_leverage_ratio_median": float(np.median([row["true_leverage_ratio"] for row in rows])), "finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean())}
    primary_causal = causal_summaries[str(config["source_layer"])]
    hypotheses = {
        "M0_analytic_jacobian_matches_autograd": {"supported": maximum_error <= 1e-5},
        "M1_rank64_captures_local_task_response": {"supported": all(block_summaries[str(block)]["ranks"]["64"]["response_cosine"]["median"] >= .90 for block in config["blocks"])},
        "M2_rank256_nearly_completes_local_task_response": {"supported": all(block_summaries[str(block)]["ranks"]["256"]["response_cosine"]["median"] >= .98 for block in config["blocks"])},
        "M3_task_response_is_mode_sparse": {"supported": all(block_summaries[str(block)]["task_prefix_k90"]["median"] <= 256 for block in config["blocks"])},
        "M4_top_subspaces_recur_across_contexts": {"supported": all(block_summaries[str(block)][key]["median"] >= 3 * random_expectation for block in config["blocks"] for key in ("u_subspace_overlap", "v_subspace_overlap"))},
        "M5_leading_singular_values_exceed_permutation_null": {"supported": all(block_summaries[str(block)]["leading_to_null_ratio"]["median"] >= 1.2 for block in config["blocks"])},
        "M6_rank64_composes_causally": {"supported": primary_causal["all_rank_64"]["gradient_cosine_median"] >= .90 and primary_causal["all_rank_64"]["finite_directional_fraction"] >= .75},
        "M7_rank16_is_not_universally_sufficient": {"supported": primary_causal["all_rank_16"]["gradient_cosine_median"] < .90 or any(block_summaries[str(block)]["ranks"]["16"]["response_cosine"]["median"] < .90 for block in config["blocks"])},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_analytic_error": maximum_error, "block_summaries": block_summaries, "causal_summaries": causal_summaries, "local_records": local_records, "causal_records": causal_records}, indent=2, allow_nan=False))
    lines = ["# Dynamic bilinear MLP Jacobian spectrum", "", "For eight held-out semantic sites, analytic state-conditioned MLP Jacobians are constructed directly from the bilinear weights and RMSNorm derivative at eight suffix blocks. Their singular modes are assessed by matrix energy, actual semantic-adjoint energy, cross-context recurrence, a channel-coefficient permutation null, and finite interventions after reinsertion into the exact suffix recurrence.", "", f"Maximum analytic-versus-autograd VJP error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Dynamic MLP Jacobians](bilin18_dynamic_mlp_jacobian_spectrum.png)", "", "| Block | Effective rank | Spectral K90 | Task-prefix K90 | Task-optimal K90 | Rank64 spectral/task-selected | Rank256 spectral/task-selected | U/V overlap | Lead/null |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for block in config["blocks"]:
        row = block_summaries[str(block)]; lines.append(f"| {block} | {row['effective_rank']['median']:.1f} | {row['spectral_k90']['median']:.0f} | {row['task_prefix_k90']['median']:.0f} | {row['task_optimal_k90']['median']:.0f} | {row['ranks']['64']['response_cosine']['median']:.3f}/{row['ranks']['64']['task_selected_response_cosine']['median']:.3f} | {row['ranks']['256']['response_cosine']['median']:.3f}/{row['ranks']['256']['task_selected_response_cosine']['median']:.3f} | {row['u_subspace_overlap']['median']:.3f}/{row['v_subspace_overlap']['median']:.3f} | {row['leading_to_null_ratio']['median']:.2f} |")
    lines += ["", "| Program | Gradient cosine | Contribution cosine | Leverage | Finite direction |", "|---|---:|---:|---:|---:|"]
    for source_layer in config.get("exploratory_source_layers", [config["source_layer"]]):
        for rank in config["causal_ranks"]:
            row = causal_summaries[str(source_layer)][f"all_rank_{rank}"]; lines.append(f"| Source {source_layer}, all sampled blocks rank {rank} | {row['gradient_cosine_median']:.3f} | {row['contribution_cosine_median']:.3f} | {row['true_leverage_ratio_median']:.3f} | {row['finite_directional_fraction']:.3f} |")
            task_row = causal_summaries[str(source_layer)][f"all_task_rank_{rank}"]; lines.append(f"| Source {source_layer}, task-selected rank {rank} | {task_row['gradient_cosine_median']:.3f} | {task_row['contribution_cosine_median']:.3f} | {task_row['true_leverage_ratio_median']:.3f} | {task_row['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrails", "", "Singular modes are computed for the current sequence position. Other positions retain exact MLP VJPs in the causal recurrence, and unsampled suffix blocks remain exact. Therefore the simultaneous truncation result is a targeted test of the analyzed operators, not yet a complete low-rank replacement of every position and block. The permutation null preserves coefficient marginals but changes which native channel receives each coefficient; it is not a full random-matrix null for the learned weights. Task-selected modes are an explicitly post-hoc, output-conditioned red-team: they measure compressibility of one backward computation, not a reusable context-independent feature basis."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained"); blocks = config["blocks"]
    for rank in config["ranks"]: axes[0].plot(blocks, [block_summaries[str(block)]["ranks"][str(rank)]["response_cosine"]["median"] for block in blocks], marker="o", label=f"rank {rank}")
    axes[0].set(xlabel="Block", ylabel="Median local response cosine", title="Task-conditioned spectral compression"); axes[0].grid(alpha=.2); axes[0].legend(ncol=2)
    x = np.arange(len(config["causal_ranks"])); width = .18
    for offset, source_layer in enumerate(config.get("exploratory_source_layers", [config["source_layer"]])): axes[1].bar(x + (offset - 1.5) * width, [causal_summaries[str(source_layer)][f"all_task_rank_{rank}"]["gradient_cosine_median"] for rank in config["causal_ranks"]], width=width, label=f"source {source_layer}")
    axes[1].set_xticks(x, [str(rank) for rank in config["causal_ranks"]]); axes[1].set(xlabel="Task-selected rank at every sampled block", ylabel="Median source-gradient cosine", title="Composition inside the exact suffix"); axes[1].grid(axis="y", alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_analytic_error": maximum_error, "block_summaries": block_summaries, "causal_summaries": causal_summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
