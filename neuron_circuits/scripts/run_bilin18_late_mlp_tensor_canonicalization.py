#!/usr/bin/env python3
"""Center and factor the spectral late-MLP operator atlas into exact circuit triads."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets, fidelity
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_late_mlp_tensor_canonicalization.json")
OUTPUT = TENSOR_ROOT / "bilin18_late_mlp_tensor_canonicalization.json"
REPORT = TENSOR_ROOT / "BILIN18_LATE_MLP_TENSOR_CANONICALIZATION.md"


@torch.inference_mode()
def gram_modes(data, count):
    values = data.cuda()
    gram = values @ values.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    count = min(count, int((eigenvalues > eigenvalues[0] * 1e-10).sum()))
    singular = eigenvalues[:count].sqrt()
    modes = ((eigenvectors[:, :count].T @ values)
             / singular[:, None].clamp_min(1e-20)).cpu()
    total = float(values.double().square().sum())
    shares = eigenvalues.double().cpu() / max(total, 1e-30)
    rank90 = int(torch.searchsorted(shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype)).item() + 1)
    return modes, singular.cpu(), {"top_energy_share": float(shares[0]), "rank90": rank90}


@torch.inference_mode()
def exact_reads(block, axes, modes):
    """Coefficient read vectors implied exactly by the bilinear weights."""
    count = len(modes); dimension = axes.shape[0]; input_rank = axes.shape[1]
    matrices = modes.view(count, input_rank, dimension).cuda()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    left_direction = axes.T.cuda() @ left.T
    right_direction = axes.T.cuda() @ right.T
    mode_down = torch.einsum("knd,dc->knc", matrices, down)
    right_coefficients = (mode_down * left_direction[None]).sum(1)
    left_coefficients = (mode_down * right_direction[None]).sum(1)
    reads = right_coefficients @ right + left_coefficients @ left
    return reads.cpu()


def weighted_median(values, weights):
    order = torch.argsort(values)
    sorted_values = values[order]; sorted_weights = weights[order]
    index = int(torch.searchsorted(sorted_weights.cumsum(0), sorted_weights.sum() / 2).item())
    return int(sorted_values[min(index, len(sorted_values) - 1)])


@torch.inference_mode()
def factor_modes(modes, singular_values, input_rank, dimension):
    matrices = modes.view(len(modes), input_rank, dimension).cuda()
    left, values, right = torch.linalg.svd(matrices, full_matrices=False)
    energy = values.double().square()
    cumulative = energy.cumsum(1) / energy.sum(1, keepdim=True).clamp_min(1e-30)
    ranks90 = (cumulative < .90).sum(1).add(1).cpu()
    weights = singular_values.double().square()
    summaries = {
        "leading_mode_inner_rank90": int(ranks90[0]),
        "leading_mode_inner_top_share": float(cumulative[0, 0]),
        "weighted_median_inner_rank90": weighted_median(ranks90, weights),
        "inner_rank90_first64": ranks90.tolist(),
        "inner_singular_values_first8_modes": values[:8].cpu().tolist(),
    }
    return left.cpu(), values.cpu(), right.cpu(), summaries


@torch.inference_mode()
def truncated_modes(left, values, right, inner_rank):
    return torch.einsum("knr,kr,krd->knd", left[:, :, :inner_rank].cuda(),
                        values[:, :inner_rank].cuda(), right[:, :inner_rank].cuda()).flatten(1).cpu()


@torch.inference_mode(False)
def main():
    raw = json.loads(CONFIG.read_text())
    base = json.loads(Path(raw["source_config"]).read_text())
    config = {**base, **{key: value for key, value in raw.items()
                        if key not in ("source_config", "hypotheses")},
              "hypotheses": raw["hypotheses"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])
    discovery_states = collect_states(model, config["atlas_discovery"], config)
    evaluation_states = {name: collect_states(model, spec, config)
                         for name, spec in config["evaluation"].items()}
    results = {}; maximum_read_error = 0.0
    for layer in config["layers"]:
        print("canonicalize layer", layer, flush=True)
        target = derivative_targets(model.blocks[layer], discovery_states[layer], axes[layer], config["batch_size"])
        mean_operator = target.mean(0)
        modes, singular, uncentered = gram_modes(target, config["operator_modes"])
        _, _, centered = gram_modes(target - mean_operator, min(len(target) - 1, config["operator_modes"] * 8))
        mean_flat = mean_operator.flatten().double()
        leading = modes[0].double()
        mean_mode_cosine = float((leading @ mean_flat).abs()
                                 / (leading.norm() * mean_flat.norm()).clamp_min(1e-30))
        reads = exact_reads(model.blocks[layer], axes[layer], modes)
        measured = target @ modes.T
        analytic = discovery_states[layer].float() @ reads.T
        read_error = float((measured - analytic).double().norm()
                           / measured.double().norm().clamp_min(1e-30))
        maximum_read_error = max(maximum_read_error, read_error)
        left, inner_values, right, factor_summary = factor_modes(
            modes, singular, config["rank"], axes[layer].shape[0])
        mode_banks = {str(rank): truncated_modes(left, inner_values, right, rank)
                      for rank in config["inner_ranks"]}
        mean_state = discovery_states[layer].double().mean(0)
        leading_read = reads[0].double()
        read_mean_state_cosine = float((leading_read @ mean_state).abs()
                                       / (leading_read.norm() * mean_state.norm()).clamp_min(1e-30))
        corpus_rows = {}
        for corpus, state_rows in evaluation_states.items():
            evaluation_target = derivative_targets(
                model.blocks[layer], state_rows[layer], axes[layer], config["batch_size"])
            coefficients = state_rows[layer].float() @ reads.T
            full_prediction = coefficients.cuda() @ modes.cuda()
            inner = {}
            for inner_rank, bank in mode_banks.items():
                prediction = coefficients.cuda() @ bank.cuda()
                inner[inner_rank] = fidelity(evaluation_target, prediction.cpu())
            corpus_rows[corpus] = {
                "mean_operator_fidelity": fidelity(
                    evaluation_target, mean_operator[None].expand_as(evaluation_target)),
                "full_rank64_exact_read_fidelity": fidelity(evaluation_target, full_prediction.cpu()),
                "inner_rank_fidelity": inner,
            }
        results[str(layer)] = {
            "read_coefficient_relative_error": read_error,
            "uncentered_spectrum": uncentered,
            "centered_spectrum": centered,
            "leading_mode_mean_operator_cosine": mean_mode_cosine,
            "leading_read_mean_state_cosine": read_mean_state_cosine,
            "factorization": factor_summary,
            "corpora": corpus_rows,
        }

    layer16 = results["16"]
    cells = [(layer, corpus, results[str(layer)]["corpora"][corpus])
             for layer in config["layers"] for corpus in config["evaluation"]]
    hypotheses = {
        "C0_exact_read_vectors_recover_mode_coefficients": maximum_read_error <= 1e-5,
        "C1_layer16_spike_is_a_common_operator": (
            layer16["corpora"]["fineweb"]["mean_operator_fidelity"] >= .60
            and layer16["leading_mode_mean_operator_cosine"] >= .95),
        "C2_layer16_variation_remains_spectrally_spiked": (
            layer16["centered_spectrum"]["top_energy_share"] >= .25),
        "C3_layer16_leading_mode_is_a_simple_matrix_circuit": (
            layer16["factorization"]["leading_mode_inner_rank90"] <= 4),
        "C4_layer16_mode_family_is_inner_low_rank": (
            layer16["factorization"]["weighted_median_inner_rank90"] <= 8),
        "C5_rank4_inner_factorization_preserves_the_atlas": all(
            row["inner_rank_fidelity"]["4"] >= row["full_rank64_exact_read_fidelity"] - .05
            for _, _, row in cells),
        "C6_layer16_has_a_compact_read_transform_write_program": all(
            layer16["corpora"][corpus]["inner_rank_fidelity"]["4"] >= .80
            for corpus in config["evaluation"]),
        "C7_earlier_layers_remain_broad_after_tensor_factorization": any(
            all(results[str(layer)]["corpora"][corpus]["inner_rank_fidelity"]["8"] < .70
                for corpus in config["evaluation"])
            for layer in (13, 14, 15)),
        "C8_exact_read_program_matches_oracle_projection": all(
            abs(row["full_rank64_exact_read_fidelity"]
                - json.loads((TENSOR_ROOT / "bilin18_late_mlp_operator_atlas.json").read_text())
                  ["results"][str(layer)]["corpora"][corpus]["budgets"]["64"]["oracle_projection_fidelity"])
            <= 1e-5 for layer, corpus, row in cells),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["C0_exact_read_vectors_recover_mode_coefficients"]["maximum_relative_error"] = maximum_read_error
    payload = {"config": raw, "hypotheses": hypothesis_rows, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Centered and factorized late-MLP operator atlas", "",
             "Each frozen spectral operator mode is converted into an exact weight-derived state read vector and then SVD-factorized between the 32D perturbation input and 1,152D output. Coefficients are never fit or chosen per example. The centered spectrum and frozen mean test whether an uncentered spike is merely a shared operator.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum exact-read coefficient error: `{maximum_read_error:.2e}`.", "",
              "| Corpus | Layer | Mean | Center top share / R90 | Mode-mean cos | Top/median inner R90 | Full64 | Inner R1 / R2 / R4 / R8 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus in config["evaluation"]:
        for layer in config["layers"]:
            row = results[str(layer)]; cell = row["corpora"][corpus]
            inner = cell["inner_rank_fidelity"]
            values = " / ".join(f"{inner[str(rank)]:.3f}" for rank in (1, 2, 4, 8))
            lines.append(f"| {corpus} | {layer} | {cell['mean_operator_fidelity']:.3f} | {row['centered_spectrum']['top_energy_share']:.3f} / {row['centered_spectrum']['rank90']} | {row['leading_mode_mean_operator_cosine']:.3f} | {row['factorization']['leading_mode_inner_rank90']} / {row['factorization']['weighted_median_inner_rank90']} | {cell['full_rank64_exact_read_fidelity']:.3f} | {values} |")
    lines += ["", "A compact triad has three independently testable parts: a state read vector, a low-rank perturbation-to-output transform, and a causal output write. Exact coefficient recovery validates the algebra but does not make the read semantic; token/context selectivity and downstream intervention transfer are separate tests."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
