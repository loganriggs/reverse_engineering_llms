#!/usr/bin/env python3
"""Canonicalize the Layer-17 tail as one basis-invariant bilinear tensor map."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_tensor_canonicalization.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_tensor_canonicalization.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_tensor_canonicalization.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_TENSOR_CANONICALIZATION.md"
FIGURE = TENSOR_ROOT / "layer17_tail_tensor_canonicalization.png"


def centered_vector_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = target - target.mean(0, keepdims=True)
    prediction = prediction - prediction.mean(0, keepdims=True)
    return float(1 - np.square(target - prediction).sum() / max(np.square(target).sum(), 1e-30))


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) - np.mean(first)
    second = np.asarray(second, dtype=np.float64) - np.mean(second)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def best_axis_match(first: np.ndarray, second: np.ndarray) -> dict:
    overlap = np.abs(np.asarray(first).T @ np.asarray(second))
    best = max(
        ((float(np.mean([overlap[i, permutation[i]] for i in range(len(permutation))])), permutation)
         for permutation in itertools.permutations(range(overlap.shape[1]))),
        key=lambda item: item[0],
    )
    values = [float(overlap[index, best[1][index]]) for index in range(overlap.shape[0])]
    return {
        "mean_cosine": float(np.mean(values)),
        "minimum_cosine": float(np.min(values)),
        "permutation": list(best[1]),
        "cosines": values,
    }


def tensor_gram(forms: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([torch.stack([(first * second).sum() for second in forms]) for first in forms])


def canonical_output(forms: list[torch.Tensor], output_basis: torch.Tensor):
    gram = tensor_gram(forms)
    values, rotation = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values, rotation = values[order], rotation[:, order]
    return values, output_basis @ rotation, rotation


def common_input(forms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    importance = sum(form @ form for form in forms)
    values, vectors = torch.linalg.eigh(importance)
    order = torch.argsort(values, descending=True)
    return values[order].clamp_min(0), vectors[:, order]


def joint_energy(forms: list[torch.Tensor], basis: torch.Tensor) -> float:
    captured = sum((basis.T @ form @ basis).square().sum() for form in forms)
    total = sum(form.square().sum() for form in forms)
    return float(captured / total.clamp_min(1e-30))


def predict_scores(
    z: torch.Tensor,
    forms: list[torch.Tensor],
    traces: torch.Tensor,
    basis: torch.Tensor,
    diagonal_only: bool = False,
) -> torch.Tensor:
    coordinates = z @ basis
    scores = []
    norm = z.square().sum(1)
    for index, form in enumerate(forms):
        restricted = basis.T @ form @ basis
        if diagonal_only:
            score = coordinates.square() @ torch.diagonal(restricted)
        else:
            score = torch.einsum("bi,ij,bj->b", coordinates, restricted, coordinates)
        scores.append(score + traces[index] * norm)
    return torch.stack(scores, dim=1)


def score_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlations = [correlation(target[:, index], prediction[:, index]) for index in range(target.shape[1])]
    return {
        "centered_vector_r2": centered_vector_r2(target, prediction),
        "median_axis_correlation": float(np.median(correlations)),
        "minimum_axis_correlation": float(np.min(correlations)),
        "axis_correlations": correlations,
    }


def causal_summary(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "median_kl": float(np.median(records["kl"])),
        "mean_kl": float(np.mean(records["kl"])),
        "ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
    }


@torch.inference_mode()
def behavioral(
    model,
    spec: dict,
    config: dict,
    tail_output: torch.Tensor,
    inverse: torch.Tensor,
    forms: list[torch.Tensor],
    traces: torch.Tensor,
    common_basis: torch.Tensor,
) -> dict[str, dict[str, float]]:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    variants = [f"rank{rank}" for rank in config["behavioral_ranks"]] + ["diagonal_only16"]
    records = {name: {field: [] for field in ("kl", "ce_delta", "top1")} for name in variants}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        product = block.mlp.Left(x) * block.mlp.Right(x)
        raw_write = block.mlp.Down(product).float()
        native_write = raw_write + block.mlp.Down_bias.float()
        native_tail_score = raw_write @ tail_output
        z = x @ inverse
        predictions = {}
        for rank in config["behavioral_ranks"]:
            predictions[f"rank{rank}"] = predict_scores(
                z, forms, traces, common_basis[:, :rank]
            )
        rank = config["diagonal_control_rank"]
        predictions["diagonal_only16"] = predict_scores(
            z, forms, traces, common_basis[:, :rank], diagonal_only=True
        )
        native_logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(pre + native_write, (dimension,))).float() / 30
        )
        native_logprob = native_logits.log_softmax(-1)
        native_probability = native_logprob.exp()
        targets = torch.as_tensor(
            rows[start : start + len(raw), np.asarray(positions) + 1].reshape(-1),
            device="cuda", dtype=torch.long,
        )
        native_ce = -native_logprob.gather(1, targets[:, None]).squeeze(1)
        native_top1 = native_logits.argmax(-1)
        for name, prediction in predictions.items():
            changed_write = native_write + (prediction - native_tail_score) @ tail_output.T
            logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30
            )
            logprob = logits.log_softmax(-1)
            kl = (native_probability * (native_logprob - logprob)).sum(1)
            ce = -logprob.gather(1, targets[:, None]).squeeze(1)
            records[name]["kl"].extend(kl.cpu().tolist())
            records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
    return {name: causal_summary(row) for name, row in records.items()}


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    tail_output = torch.from_numpy(source["tail_output_basis"][:, : config["components"]]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()

    print("FineWeb tensor metric", flush=True)
    discovery_x, _ = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(
        discovery_x, config["covariance_floor_fraction"]
    )
    raw_forms, forms, traces = [], [], []
    for component in range(config["components"]):
        coefficients = down.T @ tail_output[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        cleaned, trace = trace_clean(square @ raw @ square)
        raw_forms.append(raw)
        forms.append(cleaned)
        traces.append(trace)
    traces_tensor = torch.stack(traces)
    canonical_values, fine_canonical_output, canonical_rotation = canonical_output(forms, tail_output)
    common_values, common_basis = common_input(forms)
    energy_by_rank = {
        str(rank): joint_energy(forms, common_basis[:, :rank]) for rank in config["joint_ranks"]
    }

    arrays = {
        "tail_output_basis": tail_output.cpu().numpy(),
        "fine_canonical_output": fine_canonical_output.cpu().numpy(),
        "canonical_rotation": canonical_rotation.cpu().numpy(),
        "canonical_values": canonical_values.cpu().numpy(),
        "common_input_basis": common_basis[:, : max(config["joint_ranks"])].cpu().numpy(),
        "common_input_values": common_values[:256].cpu().numpy(),
        "square": square.cpu().numpy(),
        "inverse_square": inverse.cpu().numpy(),
        "traces": traces_tensor.cpu().numpy(),
    }

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluation", corpus, flush=True)
        x, write = collect(model, config["layer"], spec, config)
        z = x @ inverse
        native_tail_score = write @ tail_output
        score_rows = {}
        for rank in config["joint_ranks"]:
            prediction = predict_scores(z, forms, traces_tensor, common_basis[:, :rank])
            score_rows[str(rank)] = score_summary(
                native_tail_score.cpu().numpy(), prediction.cpu().numpy()
            )
        diagonal_prediction = predict_scores(
            z, forms, traces_tensor, common_basis[:, : config["diagonal_control_rank"]],
            diagonal_only=True,
        )
        diagonal = score_summary(native_tail_score.cpu().numpy(), diagonal_prediction.cpu().numpy())

        local_square, _, _ = covariance_transforms(x, config["covariance_floor_fraction"])
        local_forms = []
        for raw_form in raw_forms:
            local_clean, _ = trace_clean(local_square @ raw_form @ local_square)
            local_forms.append(local_clean)
        local_values, local_output, _ = canonical_output(local_forms, tail_output)
        match = best_axis_match(
            fine_canonical_output.cpu().numpy(), local_output.cpu().numpy()
        )
        print(" behavioral", corpus, flush=True)
        behavior = behavioral(
            model, spec, config, tail_output, inverse, forms, traces_tensor, common_basis
        )
        results[corpus] = {
            "scores": score_rows,
            "diagonal_only": diagonal,
            "canonical_output_match": match,
            "local_canonical_value_fractions": (local_values / local_values.sum()).cpu().tolist(),
            "behavioral": behavior,
        }

    hypotheses = {
        "J1_canonical_output_axes_transfer": {
            "supported": all(row["canonical_output_match"]["mean_cosine"] >= 0.80 for row in results.values()),
            "per_corpus": {key: row["canonical_output_match"]["mean_cosine"] for key, row in results.items()},
        },
        "J2_joint_rank8_reconstructs_tail_scores": {
            "supported": all(row["scores"]["8"]["median_axis_correlation"] >= 0.95 for row in results.values()),
        },
        "J3_joint_rank16_near_exact_scores": {
            "supported": all(row["scores"]["16"]["centered_vector_r2"] >= 0.98 for row in results.values()),
        },
        "J4_joint_rank16_behavioral_fidelity": {
            "supported": all(row["behavioral"]["rank16"]["median_kl"] <= 0.005 for row in results.values()),
        },
        "J5_cross_terms_are_required": {
            "supported": all(
                row["scores"]["16"]["centered_vector_r2"]
                - row["diagonal_only"]["centered_vector_r2"] >= 0.10
                for row in results.values()
            ),
        },
        "J6_joint_rank16_captures_tensor_energy": {
            "supported": energy_by_rank["16"] >= 0.95,
            "energy": energy_by_rank["16"],
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "canonical_output_value_fractions": (canonical_values / canonical_values.sum()).cpu().tolist(),
        "common_input_energy": energy_by_rank,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Basis-invariant canonicalization of the Layer-17 residual-tail tensor",
        "",
        "The stable residual tail is treated as one map from symmetric input interactions to a four-dimensional output subspace. Rotating the original tail PCs leaves this object unchanged. A FineWeb covariance metric defines cleaned forms; their 4×4 Frobenius Gram fixes tensor-canonical output coordinates, while the eigensystem of the sum of squared forms defines a shared input basis. The resulting Tucker-style program is evaluated on disjoint FineWeb and OOD states and inserted back into the native final residual path.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tail tensor canonicalization](layer17_tail_tensor_canonicalization.png)",
        "",
        "Tensor-canonical output energy fractions: **" + ", ".join(f"{value:.3f}" for value in (canonical_values / canonical_values.sum()).cpu().tolist()) + "**.",
        "",
        "| Common input rank | Joint clean tensor energy | FineWeb score R² | WikiText score R² | TinyStories score R² |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank in config["joint_ranks"]:
        lines.append(
            f"| {rank} | {energy_by_rank[str(rank)]:.3f} | "
            f"{results['fineweb']['scores'][str(rank)]['centered_vector_r2']:.3f} | "
            f"{results['wikitext']['scores'][str(rank)]['centered_vector_r2']:.3f} | "
            f"{results['tinystories']['scores'][str(rank)]['centered_vector_r2']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Canonical-axis match | Rank-16 median corr. / R² | Diagonal-only R² | Rank-16 behavioral KL |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        rank16 = row["scores"]["16"]
        lines.append(
            f"| {corpus} | {row['canonical_output_match']['mean_cosine']:.3f} | "
            f"{rank16['median_axis_correlation']:.3f} / {rank16['centered_vector_r2']:.3f} | "
            f"{row['diagonal_only']['centered_vector_r2']:.3f} | {row['behavioral']['rank16']['median_kl']:.5f} |"
        )
    lines += [
        "",
        "The common-rank program preserves the isotropic quadratic remainder exactly and compresses only the trace-cleaned interaction forms. Its output coordinates are canonical for the declared FineWeb covariance metric, not universal semantic axes. Cross-corpus axis matching tests metric stability; behavioral replacement tests whether score reconstruction survives final RMSNorm and the unembedding.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    ranks = config["joint_ranks"]
    plots[0].plot(ranks, [energy_by_rank[str(rank)] for rank in ranks], marker="o")
    plots[0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    plots[0].set_xscale("log", base=2)
    plots[0].set_xlabel("common input rank")
    plots[0].set_ylabel("joint tensor energy")
    plots[0].set_title("Shared interaction basis")
    for corpus, row in results.items():
        plots[1].plot(ranks, [row["scores"][str(rank)]["centered_vector_r2"] for rank in ranks], marker="o", label=corpus)
    plots[1].set_xscale("log", base=2)
    plots[1].set_ylim(0, 1.02)
    plots[1].set_xlabel("common input rank")
    plots[1].set_ylabel("tail-score centered R²")
    plots[1].set_title("Held-out reconstruction")
    plots[1].legend(fontsize=8)
    behavior_ranks = config["behavioral_ranks"]
    for corpus, row in results.items():
        plots[2].plot(behavior_ranks, [row["behavioral"][f"rank{rank}"]["median_kl"] for rank in behavior_ranks], marker="o", label=corpus)
    plots[2].set_xscale("log", base=2)
    plots[2].set_yscale("log")
    plots[2].set_xlabel("common input rank")
    plots[2].set_ylabel("median final-distribution KL")
    plots[2].set_title("Native tail replacement")
    plots[2].legend(fontsize=8)
    for plot in plots:
        plot.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
