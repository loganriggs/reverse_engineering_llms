#!/usr/bin/env python3
"""Semantic and finite causal validation of layer-17 full-width tensor spikes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import token_flags
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import ARRAYS as QUERY_ARRAYS
from run_bilin18_query_map_spectral_redteam import semantic_profiles
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_fullwidth_outlier_validation.json")
OUTPUT = TENSOR_ROOT / "fullwidth_outlier_validation.json"
ARRAYS = TENSOR_ROOT / "fullwidth_outlier_validation.npz"
REPORT = TENSOR_ROOT / "FULLWIDTH_OUTLIER_VALIDATION.md"
FIGURE = TENSOR_ROOT / "fullwidth_outlier_validation.png"


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first -= first.mean()
    second -= second.mean()
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    singular, full_output = output_map_eigensystem(down, left, right)
    full_output = full_output[:, : config["components"]]
    singular = singular[: config["components"]]

    query_saved = np.load(QUERY_ARRAYS)
    query_output = torch.from_numpy(query_saved["output_components"][:, : config["components"]]).cuda()
    query_canonical = torch.linalg.svdvals(full_output.T @ query_output)
    _, discovery_write = collect(model, 17, config["discovery"], config)
    discovery_write -= discovery_write.mean(0, keepdim=True)
    _, _, natural_output = torch.linalg.svd(discovery_write, full_matrices=False)
    natural_output = natural_output.T[:, : config["components"]]
    natural_canonical = torch.linalg.svdvals(full_output.T @ natural_output)

    profiles, _ = semantic_profiles(
        full_output,
        model.lm_head.weight.float()[: tokenizer.vocab_size],
        flags,
        config,
    )
    for profile in profiles:
        profile["top_tokens"] = [tokenizer.decode([index]) for index in profile.pop("top_token_ids")]
        profile["bottom_tokens"] = [tokenizer.decode([index]) for index in profile.pop("bottom_token_ids")]

    component_data = []
    arrays = {
        "singular_values": singular.cpu().numpy(),
        "output_components": full_output.cpu().numpy(),
        "query_canonical_cosines": query_canonical.cpu().numpy(),
        "natural_canonical_cosines": natural_canonical.cpu().numpy(),
    }
    for component in range(config["components"]):
        output_direction = full_output[:, component]
        coefficients = down.T @ output_direction
        matrix = left.T @ (coefficients[:, None] * right)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = torch.linalg.eigh(matrix)
        order = torch.argsort(values.abs(), descending=True)
        values = values[order]
        vectors = vectors[:, order]
        energy = values.square()
        top = vectors[:, : config["eigenspace_rank"]]
        low = vectors[:, -config["eigenspace_rank"] :]
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + component)
        random = torch.randn(left.shape[1], config["eigenspace_rank"], generator=generator, device="cuda")
        random -= top @ (top.T @ random)
        random = torch.linalg.qr(random, mode="reduced").Q
        component_data.append({
            "matrix": matrix,
            "values": values,
            "top": top,
            "low": low,
            "random": random,
            "top4_energy": float(energy[: config["eigenspace_rank"]].sum() / energy.sum()),
        })
        arrays[f"component{component}_eigenvalues"] = values[:32].cpu().numpy()
        arrays[f"component{component}_eigenvectors"] = vectors[:, :32].cpu().numpy()
    np.savez_compressed(ARRAYS, **arrays)

    evaluation_x, evaluation_write = collect(model, 17, config["evaluation"], config)
    scores = (evaluation_write @ full_output).cpu().numpy()
    source = np.load(config["evaluation"]["path"], mmap_mode="r")
    rows = source[
        config["evaluation"]["start"] : config["evaluation"]["start"] + config["evaluation"]["rows"]
    ]
    examples = {}
    selected = {}
    for component in range(config["components"]):
        centered = scores[:, component] - np.median(scores[:, component])
        indices = np.argpartition(np.abs(centered), -config["sites_per_component"])[-config["sites_per_component"] :]
        selected[component] = indices[np.argsort(np.abs(centered[indices]))[::-1]]
        examples[str(component)] = {"positive": [], "negative": []}
        for tail, indices_tail in (
            ("positive", np.argsort(scores[:, component])[-6:][::-1]),
            ("negative", np.argsort(scores[:, component])[:6]),
        ):
            for site in indices_tail:
                row_index = int(site // len(config["positions"]))
                position = config["positions"][int(site % len(config["positions"]))]
                context = rows[row_index, max(0, position - 12) : position + 1]
                examples[str(component)][tail].append({
                    "score": float(scores[site, component]),
                    "token": tokenizer.decode([int(rows[row_index, position])]),
                    "context": tokenizer.decode(np.asarray(context, dtype=np.int64).tolist()),
                })

    records = []
    dimension = model.config.n_embd
    for component, sites in selected.items():
        output_direction = full_output[:, component]
        data = component_data[component]
        for site in sites:
            row_index = int(site // len(config["positions"]))
            position = config["positions"][int(site % len(config["positions"]))]
            ids = torch.as_tensor(np.asarray(rows[row_index : row_index + 1, :128], dtype=np.int64), device="cuda")
            pre, normalized = prefix_to_layer17(model, ids)
            x = normalized[0, position].float()
            pre_site = pre[0, position].float()
            native_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)) + block.mlp.Down_bias
            native_residual = pre_site + native_write
            native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(native_residual, (dimension,))).float() / 30)
            native_logprob = native_logits.log_softmax(-1)
            native_probability = native_logprob.exp()
            native_score = float(native_write @ output_direction)
            for mode in ("top", "low", "random"):
                vectors = data[mode]
                delta = -(x @ vectors) @ vectors.T
                delta *= config["edit_norm_fraction"] * x.norm() / delta.norm().clamp_min(1e-20)
                changed_x = x + delta
                changed_write = block.mlp.Down(block.mlp.Left(changed_x) * block.mlp.Right(changed_x)) + block.mlp.Down_bias
                changed_residual = pre_site + changed_write
                changed_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(changed_residual, (dimension,))).float() / 30)
                changed_logprob = changed_logits.log_softmax(-1)
                actual = float(changed_write @ output_direction) - native_score
                matrix = data["matrix"]
                predicted = float(
                    2 * torch.einsum("i,ij,j->", delta, matrix, x)
                    + torch.einsum("i,ij,j->", delta, matrix, delta)
                )
                records.append({
                    "component": component,
                    "row": row_index,
                    "position": position,
                    "mode": mode,
                    "actual_scalar_change": actual,
                    "predicted_scalar_change": predicted,
                    "kl": float((native_probability * (native_logprob - changed_logprob)).sum()),
                    "delta_norm": float(delta.norm()),
                })

    summaries = {}
    for component in range(config["components"]):
        row = {
            "component": component,
            "singular_value": float(singular[component]),
            "semantic_profile": profiles[component],
            "top4_form_energy": component_data[component]["top4_energy"],
            "natural_score_variance": float(np.var(scores[:, component])),
            "examples": examples[str(component)],
            "modes": {},
        }
        for mode in ("top", "low", "random"):
            chosen = [record for record in records if record["component"] == component and record["mode"] == mode]
            actual = np.asarray([record["actual_scalar_change"] for record in chosen])
            predicted = np.asarray([record["predicted_scalar_change"] for record in chosen])
            row["modes"][mode] = {
                "median_absolute_scalar_change": float(np.median(np.abs(actual))),
                "median_kl": float(np.median([record["kl"] for record in chosen])),
                "prediction_correlation": correlation(actual, predicted),
            }
        summaries[str(component)] = row

    hypotheses = {
        "F1_same_as_state_query_components": {
            "supported": float(query_canonical.mean()) >= 0.75,
            "canonical_cosines": [float(value) for value in query_canonical],
        },
        "F2_same_as_natural_write_subspace": {
            "supported": float(natural_canonical.mean()) >= 0.75,
            "canonical_cosines": [float(value) for value in natural_canonical],
        },
        "F3_simple_output_roles": {
            "supported": all(row["semantic_profile"]["best_oriented_auc"] >= 0.70 for row in summaries.values())
        },
        "F4_low_rank_input_forms": {
            "supported": all(row["top4_form_energy"] >= 0.50 for row in summaries.values())
        },
        "F5_top_modes_are_specific": {
            "supported": all(
                row["modes"]["top"]["median_absolute_scalar_change"]
                >= 4 * row["modes"]["low"]["median_absolute_scalar_change"]
                for row in summaries.values()
            )
        },
        "F6_top_modes_reach_behavior": {
            "supported": sum(
                row["modes"]["top"]["median_kl"] >= 4 * row["modes"]["low"]["median_kl"]
                for row in summaries.values()
            ) >= 3
        },
        "F7_weight_formula_predicts_edits": {
            "supported": all(row["modes"]["top"]["prediction_correlation"] >= 0.99 for row in summaries.values())
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "query_canonical_cosines": [float(value) for value in query_canonical],
        "natural_canonical_cosines": [float(value) for value in natural_canonical],
        "summaries": summaries,
        "records": records,
    }, indent=2))

    lines = [
        "# Semantic and causal validation of full-width layer-17 tensor outliers",
        "",
        "These are the four strict full-width weight-spectrum spikes, selected before looking at their token roles or finite interventions. They are compared with the previous state-64 query map and empirical natural-write PCA, then tested through their exact 1,152-dimensional input interaction forms.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Mean canonical cosine with state-64 query components: **{float(query_canonical.mean()):.3f}**; with natural-write PCs: **{float(natural_canonical.mean()):.3f}**.",
        "",
        "![Full-width outlier validation](fullwidth_outlier_validation.png)",
        "",
        "| Component | Output role | AUROC | Form top-4 | Natural variance | Top/low scalar | Top/low KL | Prediction corr. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summaries.items():
        top = row["modes"]["top"]
        low = row["modes"]["low"]
        lines.append(
            f"| {int(key)+1} | {row['semantic_profile']['best_concept']} | "
            f"{row['semantic_profile']['best_oriented_auc']:.3f} | {row['top4_form_energy']:.3f} | "
            f"{row['natural_score_variance']:.2e} | "
            f"{top['median_absolute_scalar_change']/max(low['median_absolute_scalar_change'],1e-30):.1f}× | "
            f"{top['median_kl']/max(low['median_kl'],1e-30):.1f}× | {top['prediction_correlation']:.3f} |"
        )
    lines += [
        "",
        "A simple output predicate describes what an outlier tends to promote or suppress; it does not by itself describe all upstream conditions that activate it. Exact-form intervention prediction establishes the local bilinear operation, while final-logit KL establishes downstream participation rather than task-level sufficiency.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    indices = np.arange(config["components"])
    axes[0].bar(indices, [summaries[str(i)]["semantic_profile"]["best_oriented_auc"] for i in indices])
    axes[0].set_title("Output-token role")
    axes[0].set_ylabel("best oriented AUROC")
    axes[1].bar(indices, [summaries[str(i)]["top4_form_energy"] for i in indices])
    axes[1].set_title("Exact input-form compactness")
    axes[1].set_ylabel("top-four energy")
    scalar_ratios = []
    kl_ratios = []
    for i in indices:
        top = summaries[str(i)]["modes"]["top"]
        low = summaries[str(i)]["modes"]["low"]
        scalar_ratios.append(top["median_absolute_scalar_change"] / max(low["median_absolute_scalar_change"], 1e-30))
        kl_ratios.append(top["median_kl"] / max(low["median_kl"], 1e-30))
    axes[2].bar(indices - 0.18, scalar_ratios, 0.36, label="scalar")
    axes[2].bar(indices + 0.18, kl_ratios, 0.36, label="KL")
    axes[2].set_yscale("log")
    axes[2].set_title("Top vs low eigenspace")
    axes[2].set_ylabel("effect ratio")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("full-width component")
        axis.set_xticks(indices, indices + 1)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
