#!/usr/bin/env python3
"""Track reusable semantic MLP path subspaces across all model layers."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_semantic_layer0_balanced_subspaces import fit_basis
from run_bilin12_semantic_source_jvp_predictor import differentiable_attention
from run_bilin12_layer0_mlp_path_subspace import projection_fidelity, subspace_alignment, svd_summary
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_semantic_mlp_subspace_depth.json")
OUTPUT = TENSOR_ROOT / "bilin12_semantic_mlp_subspace_depth.json"
REPORT = TENSOR_ROOT / "BILIN12_SEMANTIC_MLP_SUBSPACE_DEPTH.md"
FIGURE = TENSOR_ROOT / "bilin12_semantic_mlp_subspace_depth.png"


def differentiable_mlp_trace(model, x0, position):
    x = x0
    value_bus = None
    outputs = []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (mixed.shape[-1],))
        attention, value_bus, _, _, _ = differentiable_attention(model, block, normalized, value_bus)
        pre = mixed + attention
        state = F.rms_norm(pre, (pre.shape[-1],))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
        outputs.append(mlp[:, position])
    return tuple(outputs)


def path_derivatives(model, base_x0, tangent, locations, position):
    rows = [[] for _ in model.blocks]
    function = lambda value: differentiable_mlp_trace(model, value, position)
    for location in locations:
        point = base_x0 + float(location) * tangent
        _, derivative = torch.func.jvp(function, (point,), (tangent,))
        for layer, value in enumerate(derivative):
            rows[layer].append(value.detach().float().cpu())
    return [torch.cat(values, dim=0) for values in rows]


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    locations = np.linspace(0, 1, red["path_points"])
    matrices = {layer: {} for layer in range(model.config.n_layer)}
    metadata = {}
    for record in prior["records"]:
        corpus = record["corpus"]
        row_index, start = record["row"], record["start"]
        _, target_token = (int(value) for value in record["pair"].split("->"))
        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
            tangent = changed_x0 - base_x0
        with torch.enable_grad():
            layer_matrices = path_derivatives(model, base_x0, tangent, locations, config["edit_position"])
        key = f"{corpus}|{record['pair']}"
        for layer, matrix in enumerate(layer_matrices):
            matrices[layer][key] = matrix
        metadata[key] = {"corpus": corpus, "pair": record["pair"], "family": record["family"]}
        print(corpus, record["family"], record["pair"], flush=True)

    families = sorted({row["family"] for row in metadata.values()})
    pairs = sorted({row["pair"] for row in metadata.values()})
    corpora = list(config["evaluation"])
    layer_results = {}
    for layer in range(model.config.n_layer):
        path_bases = {}
        individual_rank90 = {}
        for key, matrix in matrices[layer].items():
            summary, basis = svd_summary(matrix, [red["path_basis_rank"]])
            individual_rank90[key] = summary["rank90"]
            path_bases[key] = basis[: red["path_basis_rank"]]
        family_bases = {}
        family_rank90 = {}
        member_fidelity = {}
        for family in families:
            keys = [key for key, row in metadata.items() if row["family"] == family]
            summary, basis = fit_basis([matrices[layer][key] for key in keys], [red["family_basis_rank"]], True)
            family_bases[family] = basis
            family_rank90[family] = summary["rank90"]
            for key in keys:
                member_fidelity[key] = projection_fidelity(matrices[layer][key], basis[: red["family_basis_rank"]])
        leave_values = {}
        for pair in pairs:
            test_keys = [key for key, row in metadata.items() if row["pair"] == pair]
            family = metadata[test_keys[0]]["family"]
            train_keys = [key for key, row in metadata.items() if row["family"] == family and row["pair"] != pair]
            _, basis = fit_basis([matrices[layer][key] for key in train_keys], [red["family_basis_rank"]], True)
            for key in test_keys:
                leave_values[key] = projection_fidelity(matrices[layer][key], basis[: red["family_basis_rank"]])
        same_pair = []
        for pair in pairs:
            left, right = [f"{corpus}|{pair}" for corpus in corpora]
            _, left_basis = svd_summary(matrices[layer][left], [red["same_pair_rank"]])
            _, right_basis = svd_summary(matrices[layer][right], [red["same_pair_rank"]])
            same_pair.extend([
                projection_fidelity(matrices[layer][right], left_basis[: red["same_pair_rank"]]),
                projection_fidelity(matrices[layer][left], right_basis[: red["same_pair_rank"]]),
            ])
        same_alignment = []
        different_alignment = []
        for left, right in combinations(metadata, 2):
            if metadata[left]["pair"] == metadata[right]["pair"]:
                continue
            value = subspace_alignment(path_bases[left], path_bases[right])
            (same_alignment if metadata[left]["family"] == metadata[right]["family"] else different_alignment).append(value)
        layer_results[str(layer)] = {
            "mean_leave_pair_out": float(np.mean(list(leave_values.values()))),
            "minimum_leave_pair_out": float(min(leave_values.values())),
            "mean_same_pair_cross_corpus": float(np.mean(same_pair)),
            "minimum_member_family32": float(min(member_fidelity.values())),
            "maximum_individual_rank90": max(individual_rank90.values()),
            "family_rank90": family_rank90,
            "same_family_alignment": float(np.mean(same_alignment)),
            "different_family_alignment": float(np.mean(different_alignment)),
            "alignment_advantage": float(np.mean(same_alignment) - np.mean(different_alignment)),
            "leave_pair_out": leave_values,
        }
        print("summarized layer", layer, flush=True)

    leave_means = [layer_results[str(layer)]["mean_leave_pair_out"] for layer in range(model.config.n_layer)]
    leave_minima = [layer_results[str(layer)]["minimum_leave_pair_out"] for layer in range(model.config.n_layer)]
    best_layer = int(np.argmax(leave_means))
    neighbors = [leave_means[index] for index in (best_layer - 1, best_layer + 1) if 0 <= index < len(leave_means)]
    hypotheses = {
        "D1_leave_pair_out_transfer_improves_with_depth": {"supported": bool(max(leave_means[1:]) - leave_means[0] >= 0.10), "layer0": leave_means[0], "best_later": max(leave_means[1:]), "gain": max(leave_means[1:]) - leave_means[0]},
        "D2_some_layer_has_useful_family_transfer": {"supported": bool(max(leave_means) >= 0.50), "best": max(leave_means), "layer": best_layer},
        "D3_some_layer_has_a_robust_held_out_floor": {"supported": bool(max(leave_minima) >= 0.25), "best_minimum": max(leave_minima), "layer": int(np.argmax(leave_minima))},
        "D4_same_family_alignment_emerges": {"supported": any(layer_results[str(layer)]["alignment_advantage"] >= 0.05 for layer in range(model.config.n_layer)), "best_advantage": max(layer_results[str(layer)]["alignment_advantage"] for layer in range(model.config.n_layer)), "layer": int(np.argmax([layer_results[str(layer)]["alignment_advantage"] for layer in range(model.config.n_layer)]))},
        "D5_same_pair_cross_corpus_transfer_exceeds_family_transfer": {"supported": all(layer_results[str(layer)]["mean_same_pair_cross_corpus"] > leave_means[layer] for layer in range(model.config.n_layer)), "minimum_gap": min(layer_results[str(layer)]["mean_same_pair_cross_corpus"] - leave_means[layer] for layer in range(model.config.n_layer))},
        "D6_individual_paths_remain_low_rank_across_depth": {"supported": all(layer_results[str(layer)]["maximum_individual_rank90"] <= 16 for layer in range(model.config.n_layer)), "maximum": max(layer_results[str(layer)]["maximum_individual_rank90"] for layer in range(model.config.n_layer))},
        "D7_in_sample_family_subspaces_remain_compact": {"supported": all(layer_results[str(layer)]["minimum_member_family32"] >= 0.75 for layer in range(model.config.n_layer)), "minimum": min(layer_results[str(layer)]["minimum_member_family32"] for layer in range(model.config.n_layer))},
        "D8_semantic_reuse_has_a_distinct_best_layer": {"supported": bool(neighbors and leave_means[best_layer] - max(neighbors) >= 0.05), "best_layer": best_layer, "best": leave_means[best_layer], "neighbor_max": max(neighbors) if neighbors else None},
        "D9_late_layers_are_more_semantic_than_layer_zero": {"supported": bool(np.mean(leave_means[8:]) - leave_means[0] >= 0.10), "layer0": leave_means[0], "late_mean": float(np.mean(leave_means[8:])), "gain": float(np.mean(leave_means[8:]) - leave_means[0])},
        "D10_no_layer_has_strong_universal_family_abstraction": {"supported": bool(max(leave_minima) < 0.50), "best_minimum": max(leave_minima)},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "layers": layer_results}, indent=2, allow_nan=False))

    layers = np.arange(model.config.n_layer)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    axes[0].plot(layers, leave_means, marker="o", label="leave-pair mean")
    axes[0].plot(layers, leave_minima, marker="o", label="leave-pair minimum")
    axes[0].plot(layers, [layer_results[str(layer)]["mean_same_pair_cross_corpus"] for layer in layers], marker="o", label="same-pair cross-corpus")
    axes[0].set_title("Semantic path-subspace transfer")
    axes[1].plot(layers, [layer_results[str(layer)]["same_family_alignment"] for layer in layers], marker="o", label="same family")
    axes[1].plot(layers, [layer_results[str(layer)]["different_family_alignment"] for layer in layers], marker="o", label="different family")
    axes[1].set_title("Rank-8 path-subspace alignment")
    axes[2].plot(layers, [layer_results[str(layer)]["maximum_individual_rank90"] for layer in layers], marker="o", label="max path rank90")
    axes[2].plot(layers, [max(layer_results[str(layer)]["family_rank90"].values()) for layer in layers], marker="o", label="max family rank90")
    axes[2].set_title("Subspace ranks across depth")
    for axis in axes:
        axis.set_xlabel("layer")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Semantic MLP path-subspace transfer across depth",
        "",
        "The same 38 lexical substitutions are differentiated along 65-point input paths through every bilinear MLP. At each layer, equally weighted family bases are tested in sample, across corpora for the same lexical pair, and on lexical pairs entirely excluded from basis fitting. This asks whether pair-specific low-rank trajectories become reusable semantic computations with depth.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic subspace depth](bilin12_semantic_mlp_subspace_depth.png)",
        "",
        "| Layer | Leave-pair mean/min | Same-pair cross-corpus | Same/different alignment | Max individual rank90 | Min in-sample family32 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in range(model.config.n_layer):
        row = layer_results[str(layer)]
        lines.append(f"| {layer} | {row['mean_leave_pair_out']:.3f}/{row['minimum_leave_pair_out']:.3f} | {row['mean_same_pair_cross_corpus']:.3f} | {row['same_family_alignment']:.3f}/{row['different_family_alignment']:.3f} | {row['maximum_individual_rank90']} | {row['minimum_member_family32']:.3f} |")
    lines += [
        "",
        f"Best mean leave-pair-out transfer is **{max(leave_means):.3f}** at layer **{best_layer}**. The best worst-context floor is **{max(leave_minima):.3f}** at layer **{int(np.argmax(leave_minima))}**.",
        "",
        "Low path rank is descriptive compression. Same-pair cross-corpus transfer tests lexical recurrence; leave-pair-out family transfer is the stronger test of a reusable semantic operator. These can move differently with depth.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
