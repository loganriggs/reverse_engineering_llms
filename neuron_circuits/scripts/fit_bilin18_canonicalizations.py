#!/usr/bin/env python3
"""Fit five short descriptions of the Bilin18 suffix-ing causal tensor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_qkov_causal_edges import load_full_model


ROOT = Path("artifacts/bilin18_canonicalization")


def svd_basis(matrix, max_rank=64):
    matrix = np.asarray(matrix, dtype=np.float32)
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = min(max_rank, int((values > max(values[0], 1e-20) * 1e-9).sum()))
    basis = (vectors[:, :keep].T @ matrix) / np.sqrt(values[:keep, None].clip(1e-20))
    return basis.astype(np.float32), values


def ridge_map(features, targets, ridge=1e-3):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    scale = features.std(0).clip(1e-5)
    normalized = features / scale
    gram = normalized.T @ normalized
    regularizer = ridge * np.trace(gram) / max(len(gram), 1)
    coefficient = np.linalg.solve(gram + regularizer * np.eye(len(gram)),
                                  normalized.T @ targets)
    return scale, coefficient.astype(np.float32)


def reduced_prediction(features, scale, coefficient, rank):
    u, s, vt = np.linalg.svd(coefficient, full_matrices=False)
    keep = min(rank, len(s))
    compressed = (u[:, :keep] * s[:keep]) @ vt[:keep]
    return (features / scale) @ compressed


def metrics(target, prediction):
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    dot = (target * prediction).sum(1)
    cosine = dot / np.sqrt(np.square(target).sum(1) * np.square(prediction).sum(1)).clip(1e-20)
    fidelity = 1 - np.square(target - prediction).sum(1) / np.square(target).sum(1).clip(1e-20)
    return {"cosine_mean": float(np.nanmean(cosine)),
            "cosine_median": float(np.nanmedian(cosine)),
            "fidelity_mean": float(fidelity.mean()),
            "fidelity_median": float(np.median(fidelity))}


def main():
    arrays = np.load(ROOT / "causal_tensor.npz")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    split = np.asarray([site["split"] for site in manifest["sites"]])
    train = split == "fineweb_train"
    transfer_splits = ("fineweb_test", "wikitext103", "tinystories")
    y = arrays["target_delta"].astype(np.float32)
    source = arrays["source_delta"].astype(np.float32)
    context = arrays["input2_base"].astype(np.float32)
    input_delta = arrays["input2_delta"].astype(np.float32)

    # 1: exact native coordinate values, ranked with weights only.
    model = load_full_model("cuda" if torch.cuda.is_available() else "cpu")
    units = torch.as_tensor(manifest["units"], device=model.blocks[1].mlp.Down.weight.device)
    down = model.blocks[1].mlp.Down.weight.detach().float()[:, units]
    left2 = model.blocks[2].mlp.Left.weight.detach().float()
    right2 = model.blocks[2].mlp.Right.weight.detach().float()
    down2 = model.blocks[2].mlp.Down.weight.detach().float().cpu().numpy()
    weight_score = ((left2 @ down).square() + (right2 @ down).square()).mean(1).sqrt()
    coordinate_ranking = torch.argsort(weight_score, descending=True).cpu().numpy()

    # 2: ordinary target-delta SVD; describes the shared causal output subspace.
    delta_basis, delta_spectrum = svd_basis(y[train], 64)

    # An individual-source tensor basis tests whether preserving the source mode
    # makes the shared destination description shorter.
    individual = arrays["individual_target_delta"].astype(np.float32)
    individual_basis, individual_spectrum = svd_basis(individual[train].reshape(-1, y.shape[1]), 64)

    # 3: map four source changes directly to the target population.
    source_scale, source_map = ridge_map(source[train], y[train])

    # 4: context-gated tensor-product features source_i * [1, context PCs].
    context_mean = context[train].mean(0)
    context_basis, _ = svd_basis(context[train] - context_mean, 8)
    context_score = (context - context_mean) @ context_basis.T
    context_scale = context_score[train].std(0).clip(1e-5)
    context_score /= context_scale
    gated = np.concatenate([source] + [source * context_score[:, j:j + 1]
                                       for j in range(context_score.shape[1])], axis=1)
    gated_scale, gated_map = ridge_map(gated[train], y[train], ridge=1e-2)

    # 5: compress the post-attention layer-2 input change, then use the exact
    # known bilinear weights and base factors to lift it back to product space.
    input_basis, input_spectrum = svd_basis(input_delta[train], 64)
    left_weight = left2.cpu().numpy()
    right_weight = right2.cpu().numpy()
    left_base = arrays["left2_base"].astype(np.float32)
    right_base = arrays["right2_base"].astype(np.float32)
    # The full block boundary includes the residual/attention branch as well as
    # the MLP write. This is the candidate CPU-like intermediate state.
    pre2_delta = (arrays["mixed2_delta"].astype(np.float32) +
                  arrays["attention2_delta"].astype(np.float32))
    # Use the actually observed bf16 block-state difference. Reconstructing it
    # algebraically as pre_delta + Down(product_delta) is not numerically exact
    # in this checkpoint because very large bilinear terms cancel in bf16.
    state2_delta = arrays["state2_delta"].astype(np.float32)
    algebraic_state2_delta = pre2_delta + y @ down2.T
    state2_basis, state2_spectrum = svd_basis(state2_delta[train], 64)

    selected_indices = []
    for name in transfer_splits:
        indices = np.flatnonzero(split == name)
        selected_indices.extend(indices[np.linspace(0, len(indices) - 1, 24).round().astype(int)])
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    restorations = {"indices": selected_indices}
    analysis = {"methods": {}, "splits": {}, "selected_finite_sites": {
        name: int((split[selected_indices] == name).sum()) for name in transfer_splits}}

    method_specs = {
        "weight_coordinates": (32, 128, 512, 2048),
        "delta_svd": (1, 4, 16, 64),
        "individual_tensor_svd": (1, 4, 16, 64),
        "source_reduced_rank": (1, 2, 4),
        "context_gated_rank": (1, 4, 16, 32),
        "attention_input_tensor": (1, 4, 16, 64),
        "exact_input_bilinear": (1152,),
        "layer2_state_svd": (1, 4, 16, 64),
        "exact_layer2_state_delta_replay": (1152,),
        "exact_layer2_state_overwrite": (1152,),
    }
    for method, ranks in method_specs.items():
        analysis["methods"][method] = {"ranks": list(ranks), "predictive": {}}
        for rank in ranks:
            if method == "weight_coordinates":
                prediction = np.zeros_like(y)
                chosen = coordinate_ranking[:rank]
                prediction[:, chosen] = y[:, chosen]
            elif method == "delta_svd":
                basis = delta_basis[:rank]
                prediction = (y @ basis.T) @ basis
            elif method == "individual_tensor_svd":
                basis = individual_basis[:rank]
                prediction = (y @ basis.T) @ basis
            elif method == "source_reduced_rank":
                prediction = reduced_prediction(source, source_scale, source_map, rank)
            elif method == "context_gated_rank":
                prediction = reduced_prediction(gated, gated_scale, gated_map, rank)
            elif method == "attention_input_tensor":
                basis = input_basis[:rank]
                reconstructed = (input_delta @ basis.T) @ basis
                delta_left = reconstructed @ left_weight.T
                delta_right = reconstructed @ right_weight.T
                prediction = (delta_left * right_base + left_base * delta_right -
                              delta_left * delta_right)
            elif method == "exact_input_bilinear":
                delta_left = input_delta @ left_weight.T
                delta_right = input_delta @ right_weight.T
                prediction = (delta_left * right_base + left_base * delta_right -
                              delta_left * delta_right)
            elif method == "layer2_state_svd":
                basis = state2_basis[:rank]
                prediction = (state2_delta @ basis.T) @ basis
            elif method in {"exact_layer2_state_delta_replay", "exact_layer2_state_overwrite"}:
                prediction = state2_delta
            else:
                raise AssertionError(method)
            key = f"{method}__{rank}"
            restorations[key] = prediction[selected_indices].astype(np.float16)
            metric_target = state2_delta if method in {
                "layer2_state_svd", "exact_layer2_state_delta_replay",
                "exact_layer2_state_overwrite"} else y
            analysis["methods"][method]["predictive"][str(rank)] = {
                name: metrics(metric_target[split == name], prediction[split == name])
                for name in transfer_splits
            }

    # Descriptive diagnostics for the exact attention path before RMSNorm.
    mixed = arrays["mixed2_delta"].astype(np.float32)
    attention = arrays["attention2_delta"].astype(np.float32)
    analysis["attention_route"] = {
        name: {
            "attention_to_mixed_norm_ratio_median": float(np.median(
                np.linalg.norm(attention[split == name], axis=1) /
                np.linalg.norm(mixed[split == name], axis=1).clip(1e-20))),
            "attention_mixed_cosine_median": float(np.median(
                (attention[split == name] * mixed[split == name]).sum(1) /
                (np.linalg.norm(attention[split == name], axis=1) *
                 np.linalg.norm(mixed[split == name], axis=1)).clip(1e-20))),
        } for name in transfer_splits
    }
    analysis["bf16_algebraic_state_error"] = {
        name: float(np.median(np.linalg.norm(
            algebraic_state2_delta[split == name] - state2_delta[split == name], axis=1) /
            np.linalg.norm(state2_delta[split == name], axis=1).clip(1e-20)))
        for name in transfer_splits
    }
    analysis["spectra"] = {
        "delta_top16_fraction": float(delta_spectrum[:16].sum() / delta_spectrum.clip(0).sum()),
        "individual_top16_fraction": float(individual_spectrum[:16].sum() /
                                            individual_spectrum.clip(0).sum()),
        "input_top16_fraction": float(input_spectrum[:16].sum() / input_spectrum.clip(0).sum()),
        "state2_top16_fraction": float(state2_spectrum[:16].sum() /
                                        state2_spectrum.clip(0).sum()),
    }
    np.savez_compressed(ROOT / "restoration_vectors.npz", **restorations)
    (ROOT / "fit_analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps({
        method: {rank: round(np.mean([
            analysis["methods"][method]["predictive"][rank][name]["fidelity_median"]
            for name in transfer_splits]), 3)
                 for rank in analysis["methods"][method]["predictive"]}
        for method in method_specs
    }, indent=2))


if __name__ == "__main__":
    main()
