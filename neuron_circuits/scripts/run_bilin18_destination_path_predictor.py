#!/usr/bin/env python3
"""Prospectively predict destination compatibility from exact tensor-network paths.

The predictor never uses a candidate's finite intervention outcome.  Candidate
operator differences are contracted through the checkpoint's analytic JVP.
The exact tangent is then compared with a readable first-order path program:
the attention-free path plus one path for each possible single downstream
attention insertion.  Small finite replacements are outcomes used only after
the analytic predictions have been formed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_attention_layer_path_program import (
    mlp_post_jvp,
    sequence_rms_jvp,
    two_qk_attention_jvp,
)
from run_bilin18_destination_fisher_geometry import cosine_gram
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import (
    apply_function,
    build_function_bank,
    intervention_logits,
)
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_destination_path_predictor.json")
OUTPUT = TENSOR_ROOT / "bilin18_destination_path_predictor.json"
TENSORS = TENSOR_ROOT / "bilin18_destination_path_predictor.pt"
REPORT = TENSOR_ROOT / "BILIN18_DESTINATION_PATH_PREDICTOR.md"


def median_angle_partner(gram: np.ndarray, norms: np.ndarray, target: int) -> int:
    candidates = np.asarray([index for index in range(len(norms)) if index != target])
    cosine = np.abs(gram[target, candidates] / (norms[target] * norms[candidates]))
    median = np.median(cosine)
    return int(candidates[np.argmin(np.abs(cosine - median))])


def fidelity(target2: float, prediction2: float, dot: float) -> float:
    return float(1 - (target2 + prediction2 - 2 * dot) / max(target2, 1e-30))


def cosine(target2: float, prediction2: float, dot: float) -> float:
    return float(dot / np.sqrt(max(target2 * prediction2, 1e-30)))


def response(model, final_state, tangents):
    """Return candidate Fisher response and Gram at selected output sites."""
    normalized = F.rms_norm(final_state, (final_state.shape[-1],))
    raw = F.linear(normalized, model.lm_head.weight.float())
    logits = 30 * torch.tanh(raw / 30)
    probability = F.softmax(logits, -1)
    normalized_tangent = sequence_rms_jvp(final_state, tangents)
    raw_tangent = F.linear(normalized_tangent, model.lm_head.weight.float())
    cap = 1 - torch.tanh(raw / 30).square()
    logit_tangent = raw_tangent * cap[:, None]
    centered = logit_tangent - (logit_tangent * probability[:, None]).sum(-1, keepdim=True)
    weighted = centered * probability.sqrt()[:, None]
    flat = weighted.permute(1, 0, 2, 3).flatten(1).double()
    return logits, flat, flat @ flat.T


@torch.inference_mode()
def analytic_candidate_paths(model, ids, target_row, partner_row, partner_scale,
                             controls, bank, positions, max_order=1,
                             capture_layers=None, return_cache=False):
    """Propagate all candidate operator deltas exactly and by first-order paths."""
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    x0 = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x = x0
    bus = None
    later = []
    target_capture = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0].float() * x + block.lambdas[1].float() * x0
        attention_input = F.rms_norm(mixed, (dimension,))
        bus_before = bus
        details = attention_details(block.attn, attention_input, bus, model.config, model.tt)
        bus = details["value_bus"]
        pre = mixed + details["write"].float()
        state = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(state) * block.mlp.Right(state)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        row = {
            "layer": layer,
            "mixed": mixed,
            "attention_input": attention_input,
            "bus": bus_before,
            "pre": pre,
            "state": state,
        }
        if layer == target_row["layer"]:
            batch, length = ids.shape
            pattern = details["pattern"][:, target_row["head"]]
            pooled_state = torch.einsum("bqk,bkd->bqd", pattern, attention_input)
            flat_bus = bus.reshape(batch, length, dimension)
            pooled_bus = torch.einsum("bqk,bkd->bqd", pattern, flat_bus)
            native = apply_function(target_row, pooled_state, pooled_bus, heads)
            replacements = [partner_scale * apply_function(
                partner_row, pooled_state, pooled_bus, heads,
            )]
            for control in controls:
                other = apply_function(bank[control["other"]], pooled_state, pooled_bus, heads)
                replacements.append(
                    control["target_coefficient"] * native
                    + control["other_coefficient"] * other
                )
            delta = torch.stack(replacements, 1) - native[:, None]
            initial = mlp_post_jvp(block, pre, state, delta)
            target_capture = (initial, bus)
        elif layer > target_row["layer"]:
            later.append(row)
    if target_capture is None:
        raise RuntimeError("target layer not encountered")

    capture_layers = set() if capture_layers is None else set(capture_layers)
    full = target_capture[0]
    captured_tangents = {}
    if target_row["layer"] in capture_layers:
        captured_tangents[target_row["layer"]] = full
    # Keep paths as a streamed list.  Flattening path x candidate before the
    # 4608-channel MLP creates a >1 GiB temporary on 16 GiB GPUs even though
    # the path states themselves are modest.
    paths = [full]
    higher_orders = [torch.zeros_like(full) for _ in range(max(0, max_order - 1))]
    path_labels = ["direct"]
    transfer_stats = []
    maximum_attention_forward_error = 0.0
    for downstream_index, row in enumerate(later, start=1):
        layer = row["layer"]
        block = model.blocks[layer]

        full_mixed = block.lambdas[0].float() * full
        full_input = sequence_rms_jvp(row["mixed"], full_mixed)
        full_attention, error = two_qk_attention_jvp(
            block.attn, row["attention_input"], row["bus"], full_input,
        )
        maximum_attention_forward_error = max(maximum_attention_forward_error, error)
        full_next = mlp_post_jvp(
            block, row["pre"], row["state"], full_mixed + full_attention,
        )
        identity_branch = mlp_post_jvp(
            block, row["pre"], row["state"], full_mixed,
        )
        attention_branch = full_next - identity_branch
        incoming2 = float(full.square().sum())
        identity2 = float(identity_branch.square().sum())
        attention2 = float(attention_branch.square().sum())
        outgoing2 = float(full_next.square().sum())
        branch_dot = float((identity_branch * attention_branch).sum())
        transfer_stats.append({
            "layer": layer,
            "gain": float(np.sqrt(outgoing2 / max(incoming2, 1e-30))),
            "identity_norm": float(np.sqrt(identity2)),
            "attention_norm": float(np.sqrt(attention2)),
            "outgoing_norm": float(np.sqrt(outgoing2)),
            "attention_to_identity": float(np.sqrt(attention2 / max(identity2, 1e-30))),
            "branch_cosine": float(branch_dot / np.sqrt(max(identity2 * attention2, 1e-30))),
            "cancellation_condition": float(
                (np.sqrt(identity2) + np.sqrt(attention2)) / np.sqrt(max(outgoing2, 1e-30))
            ),
        })
        full = full_next
        if layer in capture_layers:
            captured_tangents[layer] = full

        if max_order > 1:
            old_order_one = sum(paths[1:], torch.zeros_like(paths[0]))
            old_orders = [paths[0], old_order_one] + higher_orders
            next_higher = []
            for order in range(2, max_order + 1):
                if order > downstream_index:
                    next_higher.append(torch.zeros_like(full))
                    continue
                own_mixed = block.lambdas[0].float() * old_orders[order]
                source_mixed = block.lambdas[0].float() * old_orders[order - 1]
                source_input = sequence_rms_jvp(row["mixed"], source_mixed)
                routed, error = two_qk_attention_jvp(
                    block.attn, row["attention_input"], row["bus"], source_input,
                )
                maximum_attention_forward_error = max(maximum_attention_forward_error, error)
                next_higher.append(mlp_post_jvp(
                    block, row["pre"], row["state"], own_mixed + routed,
                ))
            higher_orders = next_higher

        propagated = [
            mlp_post_jvp(
                block, row["pre"], row["state"], block.lambdas[0].float() * path,
            )
            for path in paths
        ]
        direct_mixed = block.lambdas[0].float() * paths[0]
        direct_input = sequence_rms_jvp(row["mixed"], direct_mixed)
        inserted_attention, error = two_qk_attention_jvp(
            block.attn, row["attention_input"], row["bus"], direct_input,
        )
        maximum_attention_forward_error = max(maximum_attention_forward_error, error)
        inserted = mlp_post_jvp(
            block, row["pre"], row["state"], inserted_attention,
        )
        paths = propagated + [inserted]
        path_labels.append(f"A{layer}")

    exact_orders = [paths[0], sum(paths[1:], torch.zeros_like(paths[0]))] + higher_orders
    order_sum = sum(exact_orders[:max_order + 1], torch.zeros_like(full))
    order_recomposition_relative_error = float(
        (order_sum - full).norm() / full.norm().clamp_min(1e-30)
    )
    return {
        "final_state": x[:, positions].float(),
        "full": full[:, :, positions].float(),
        "paths": torch.stack(paths, 1)[:, :, :, positions].float(),
        "orders": torch.stack(exact_orders[:max_order + 1], 1)[:, :, :, positions].float(),
        "path_labels": path_labels,
        "transfer_stats": transfer_stats,
        "captured_tangents": captured_tangents,
        "later_cache": later if return_cache else None,
        "order_recomposition_relative_error": order_recomposition_relative_error,
        "attention_forward_error": maximum_attention_forward_error,
    }


@torch.inference_mode()
def propagate_candidate_tangent(model, tangent, later_cache, start_layer):
    """Propagate a tangent after ``start_layer`` through the exact native suffix."""
    maximum_forward_error = 0.0
    current = tangent
    for row in later_cache:
        if row["layer"] <= start_layer:
            continue
        block = model.blocks[row["layer"]]
        mixed = block.lambdas[0].float() * current
        attention_input = sequence_rms_jvp(row["mixed"], mixed)
        attention, error = two_qk_attention_jvp(
            block.attn, row["attention_input"], row["bus"], attention_input,
        )
        maximum_forward_error = max(maximum_forward_error, error)
        current = mlp_post_jvp(
            block, row["pre"], row["state"], mixed + attention,
        )
    return current, maximum_forward_error


def geometry_correlation(first, second):
    triangle = np.triu_indices(len(first), 1)
    return spearman(cosine_gram(first)[triangle], cosine_gram(second)[triangle])


def summarize_accumulator(accumulator, raw_delta_norm2):
    full = accumulator["full_gram"]
    finite = accumulator["finite_gram"]
    rows = {}
    for name in ("identity", "first_order"):
        target2 = accumulator["full_norm2"]
        prediction2 = accumulator[f"{name}_norm2"]
        dot = accumulator[f"{name}_dot"]
        rows[name] = {
            "response_fidelity": fidelity(target2, prediction2, dot),
            "response_cosine": cosine(target2, prediction2, dot),
            "fisher_geometry_correlation": geometry_correlation(
                full, accumulator[f"{name}_gram"],
            ),
            "diagonal_rank_correlation": spearman(
                np.diag(full), np.diag(accumulator[f"{name}_gram"]),
            ),
        }
    rows["analytic_vs_finite"] = {
        "fisher_geometry_correlation": geometry_correlation(full, finite),
        "diagonal_rank_correlation": spearman(np.diag(full), np.diag(finite)),
        "finite_kl_rank_correlation": spearman(np.diag(full), accumulator["finite_kl"]),
        "raw_norm_finite_kl_rank_correlation": spearman(raw_delta_norm2, accumulator["finite_kl"]),
        "tangent_response_fidelity": fidelity(
            accumulator["finite_norm2"], accumulator["full_norm2"],
            accumulator["finite_full_dot"],
        ),
        "tangent_response_cosine": cosine(
            accumulator["finite_norm2"], accumulator["full_norm2"],
            accumulator["finite_full_dot"],
        ),
    }
    energy = np.asarray(accumulator["path_energy"])
    rows["path_energy"] = dict(zip(accumulator["path_labels"], (energy / max(energy.sum(), 1e-30)).tolist()))
    return rows


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bank, raw_gram, norms = build_function_bank(model, config["candidate_layers"])
    labels = [row["label"] for row in bank]
    corpus = np.load(config["corpus"], mmap_mode="r")
    rng = np.random.default_rng(config["evaluation_random_seed"])
    indices = np.sort(rng.choice(
        np.arange(config["evaluation_index_low"], config["evaluation_index_high"]),
        config["evaluation_rows"], replace=False,
    ))
    raw = np.asarray(corpus[indices, :config["sequence_length"]], dtype=np.int64)
    halves = {"a": raw[:len(raw) // 2], "b": raw[len(raw) // 2:]}
    positions = torch.as_tensor(config["positions"], device="cuda")
    target_results = {}
    saved = {}
    maximum_forward_error = 0.0

    for target_layer, target_head in config["targets"]:
        target_label = f"L{target_layer}H{target_head}"
        target = labels.index(target_label)
        partner = median_angle_partner(raw_gram, norms, target)
        partner_scale, controls, matched_cosine = complete_controls(
            raw_gram, norms, target, partner,
        )
        candidate_labels = [labels[partner]] + [labels[row["other"]] for row in controls]
        coefficients = []
        partner_coefficient = np.zeros(len(bank)); partner_coefficient[target] = -1
        partner_coefficient[partner] = partner_scale; coefficients.append(partner_coefficient)
        for control in controls:
            coefficient = np.zeros(len(bank)); coefficient[target] = control["target_coefficient"] - 1
            coefficient[control["other"]] = control["other_coefficient"]
            coefficients.append(coefficient)
        coefficients = np.stack(coefficients)
        raw_delta_gram = coefficients @ raw_gram @ coefficients.T
        raw_delta_norm2 = np.diag(raw_delta_gram)
        target_rows = {}
        for half, panel in halves.items():
            print("destination path predictor", target_label, half, flush=True)
            accumulator = None
            for row_index, row in enumerate(panel):
                ids = torch.as_tensor(row, device="cuda")[None]
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], partner_scale,
                    controls, bank, positions,
                )
                maximum_forward_error = max(
                    maximum_forward_error, analytic["attention_forward_error"],
                )
                native_logits, full_response, full_gram = response(
                    model, analytic["final_state"], analytic["full"],
                )
                identity = analytic["paths"][:, 0]
                first_order = analytic["paths"].sum(1)
                _, identity_response, identity_gram = response(
                    model, analytic["final_state"], identity,
                )
                _, first_response, first_gram = response(
                    model, analytic["final_state"], first_order,
                )
                path_energy = []
                for path_index in range(analytic["paths"].shape[1]):
                    _, path_response, _ = response(
                        model, analytic["final_state"], analytic["paths"][:, path_index],
                    )
                    path_energy.append(float(path_response.square().sum()))

                finite_logits, _ = intervention_logits(
                    model, ids, bank[target], bank[partner], partner_scale,
                    controls, bank, positions, config["finite_scale"],
                )
                finite_delta = (finite_logits[2:] - finite_logits[0:1]) / config["finite_scale"]
                probability = F.softmax(finite_logits[0].float(), -1)
                finite_centered = finite_delta - (
                    finite_delta * probability[None]
                ).sum(-1, keepdim=True)
                finite_response = (
                    finite_centered * probability.sqrt()[None]
                ).flatten(1).double()
                finite_gram = finite_response @ finite_response.T
                # At a 1% edit the true KL is small enough that subtracting two
                # float32 log-softmax values measurably scrambles candidate
                # ranks for early targets.  Compute the validation outcome in
                # float64; the predictor itself remains unchanged.
                logp = F.log_softmax(finite_logits[0].double(), -1)
                finite_kl = (
                    logp.exp()[None] * (logp[None] - F.log_softmax(finite_logits[2:].double(), -1))
                ).flatten(1).sum(-1).double()

                if accumulator is None:
                    count = len(candidate_labels)
                    accumulator = {
                        "path_labels": analytic["path_labels"],
                        "full_gram": np.zeros((count, count)),
                        "identity_gram": np.zeros((count, count)),
                        "first_order_gram": np.zeros((count, count)),
                        "finite_gram": np.zeros((count, count)),
                        "finite_kl": np.zeros(count),
                        "full_norm2": 0.0, "identity_norm2": 0.0,
                        "first_order_norm2": 0.0, "finite_norm2": 0.0,
                        "identity_dot": 0.0, "first_order_dot": 0.0,
                        "finite_full_dot": 0.0,
                        "path_energy": np.zeros(len(analytic["path_labels"])),
                    }
                accumulator["full_gram"] += full_gram.cpu().numpy()
                accumulator["identity_gram"] += identity_gram.cpu().numpy()
                accumulator["first_order_gram"] += first_gram.cpu().numpy()
                accumulator["finite_gram"] += finite_gram.cpu().numpy()
                accumulator["finite_kl"] += finite_kl.cpu().numpy()
                accumulator["full_norm2"] += float(full_response.square().sum())
                accumulator["identity_norm2"] += float(identity_response.square().sum())
                accumulator["first_order_norm2"] += float(first_response.square().sum())
                accumulator["finite_norm2"] += float(finite_response.square().sum())
                accumulator["identity_dot"] += float((identity_response * full_response).sum())
                accumulator["first_order_dot"] += float((first_response * full_response).sum())
                accumulator["finite_full_dot"] += float((finite_response * full_response).sum())
                accumulator["path_energy"] += np.asarray(path_energy)
                print(" ", row_index + 1, "/", len(panel), flush=True)
            target_rows[half] = summarize_accumulator(accumulator, raw_delta_norm2)
            saved[f"{target_label}_{half}_analytic_gram"] = torch.from_numpy(accumulator["full_gram"])
            saved[f"{target_label}_{half}_finite_gram"] = torch.from_numpy(accumulator["finite_gram"])
        common = sorted(set(target_rows["a"]["path_energy"]) & set(target_rows["b"]["path_energy"]))
        path_stability = spearman(
            [target_rows["a"]["path_energy"][key] for key in common],
            [target_rows["b"]["path_energy"][key] for key in common],
        )
        target_results[target_label] = {
            "partner": labels[partner],
            "matched_abs_cosine": matched_cosine,
            "candidate_labels": candidate_labels,
            "path_energy_split_stability": path_stability,
            "halves": target_rows,
        }

    every = [row for target in target_results.values() for row in target["halves"].values()]
    p1 = all(
        row["analytic_vs_finite"]["fisher_geometry_correlation"] >= .95
        and row["analytic_vs_finite"]["diagonal_rank_correlation"] >= .98
        for row in every
    )
    p2 = all(
        row["first_order"]["response_fidelity"] >= .90
        and row["first_order"]["fisher_geometry_correlation"] >= .90
        for row in every
    )
    p3 = any(
        target["halves"]["a"]["identity"]["response_fidelity"] < .90
        and target["halves"]["b"]["identity"]["response_fidelity"] < .90
        for target in target_results.values()
    )
    p4 = all(target["path_energy_split_stability"] >= .80 for target in target_results.values())
    p5 = all(
        row["analytic_vs_finite"]["finite_kl_rank_correlation"]
        >= row["analytic_vs_finite"]["raw_norm_finite_kl_rank_correlation"] + .30
        for row in every
    )
    hypotheses = {
        "P1_analytic_tensor_jvp_predicts_finite_candidate_geometry": {"supported": p1},
        "P2_first_order_attention_path_program_is_sufficient": {"supported": p2},
        "P3_identity_only_is_not_sufficient": {"supported": p3},
        "P4_path_importance_is_split_stable": {"supported": p4},
        "P5_tensor_predictor_beats_raw_operator_geometry": {"supported": p5},
        "P6_prediction_is_not_one_target_specific": {"supported": p1 and p2},
        "attention_jvp_forward_invariant": {
            "supported": maximum_forward_error <= 1e-5,
            "maximum_relative_error": maximum_forward_error,
        },
    }
    for item in hypotheses.values():
        item["supported"] = bool(item["supported"])
    result = {
        "config": config,
        "evaluation_indices": indices.tolist(),
        "hypotheses": hypotheses,
        "targets": target_results,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    torch.save(saved, TENSORS)

    lines = [
        "# Prospective tensor-path predictor of destination compatibility", "",
        "Three previously untested destinations were fixed before running. Candidate operator differences are computed from weights and the native target router, then propagated through the exact analytic checkpoint JVP. Finite 1% replacements are used only as held-out outcomes. The readable approximation contains the attention-free path plus every path with exactly one downstream attention insertion; it omits all paths containing two or more downstream attention operations.", "",
        "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "## Results", "", "| Target / half | Analytic→finite geometry / diagonal | Analytic→KL | Raw→KL | Identity fidelity | First-order fidelity / geometry |", "|---|---:|---:|---:|---:|---:|"]
    for target, target_row in target_results.items():
        for half, row in target_row["halves"].items():
            af = row["analytic_vs_finite"]
            lines.append(
                f"| {target} / {half} | {af['fisher_geometry_correlation']:.3f} / {af['diagonal_rank_correlation']:.3f} | "
                f"{af['finite_kl_rank_correlation']:.3f} | {af['raw_norm_finite_kl_rank_correlation']:.3f} | "
                f"{row['identity']['response_fidelity']:.3f} | {row['first_order']['response_fidelity']:.3f} / {row['first_order']['fisher_geometry_correlation']:.3f} |"
            )
        lines.append(f"| {target} path-energy split stability | {target_row['path_energy_split_stability']:.3f} | — | — | — | — |")
    lines += [
        "", "## Interpretation guardrails", "",
        "The exact analytic JVP is a local tensor-network calculation, not a semantic explanation. The first-order path program is interpretable at the level of downstream attention use, but success would not name the variables transported by those paths. Failure would specifically imply that compositions containing two or more downstream attention operations are behaviorally necessary; it would not invalidate the exact analytic predictor. Candidate replacements retain the target QK router and differ only in the value-side operator. Results therefore concern typed operator substitution, not whole-head interchangeability.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
