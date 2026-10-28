#!/usr/bin/env python3
"""Balance semantic edit paths before fitting reusable layer-0 MLP subspaces."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_his_her_layer0_mlp_path import layer0_components, path_row
from run_bilin12_layer0_mlp_path_subspace import projection_fidelity, subspace_alignment, svd_summary, vector_projection_fidelity
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_semantic_layer0_balanced_subspaces.json")
OUTPUT = TENSOR_ROOT / "bilin12_semantic_layer0_balanced_subspaces.json"
REPORT = TENSOR_ROOT / "BILIN12_SEMANTIC_LAYER0_BALANCED_SUBSPACES.md"
FIGURE = TENSOR_ROOT / "bilin12_semantic_layer0_balanced_subspaces.png"


def normalized_stack(matrices):
    return torch.cat([matrix / matrix.norm().clamp_min(1e-30) for matrix in matrices], dim=0)


def fit_basis(matrices, ranks, balanced=True):
    matrix = normalized_stack(matrices) if balanced else torch.cat(matrices, dim=0)
    summary, basis = svd_summary(matrix, ranks)
    return summary, basis


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
    locations = np.linspace(0, 1, red["dense_points"])
    matrices = {}
    finite = {}
    metadata = {}
    path_bases = {}
    individual = {}

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
            base_components = layer0_components(model, base_x0, config["edit_position"])
            changed_components = layer0_components(model, changed_x0, config["edit_position"])
        with torch.enable_grad():
            path = [path_row(model, base_x0, tangent, float(location), config["edit_position"]) for location in locations]
        matrix = torch.cat([row["derivative"]["mlp"] for row in path], dim=0)
        summary, basis = svd_summary(matrix, red["ranks"])
        key = f"{corpus}|{record['pair']}"
        matrices[key] = matrix
        finite[key] = changed_components[7].float() - base_components[7].float()
        metadata[key] = {"corpus": corpus, "pair": record["pair"], "family": record["family"]}
        path_bases[key] = basis[:8]
        individual[key] = summary
        print(corpus, record["family"], record["pair"], flush=True)

    families = sorted({row["family"] for row in metadata.values()})
    family_results = {}
    family_bases = {}
    for family in families:
        keys = [key for key, row in metadata.items() if row["family"] == family]
        summary, basis = fit_basis([matrices[key] for key in keys], red["ranks"], True)
        family_bases[family] = basis
        family_results[family] = {"summary": summary, "members": keys}
    global_balanced, global_balanced_basis = fit_basis(list(matrices.values()), red["ranks"], True)
    global_unbalanced, global_unbalanced_basis = fit_basis(list(matrices.values()), red["ranks"], False)

    member_rows = {}
    for key, matrix in matrices.items():
        family = metadata[key]["family"]
        member_rows[key] = {
            "individual_rank90": individual[key]["rank90"],
            "family32": projection_fidelity(matrix, family_bases[family][:32]),
            "global_balanced32": projection_fidelity(matrix, global_balanced_basis[:32]),
            "global_balanced64": projection_fidelity(matrix, global_balanced_basis[:64]),
            "global_unbalanced32": projection_fidelity(matrix, global_unbalanced_basis[:32]),
            "family_finite32": vector_projection_fidelity(finite[key], family_bases[family][:32]),
        }

    leave_pair_out = {}
    for pair in sorted({row["pair"] for row in metadata.values()}):
        test_keys = [key for key, row in metadata.items() if row["pair"] == pair]
        family = metadata[test_keys[0]]["family"]
        train_keys = [key for key, row in metadata.items() if row["family"] == family and row["pair"] != pair]
        _, basis = fit_basis([matrices[key] for key in train_keys], [red["transfer_rank"]], True)
        leave_pair_out[pair] = {key: projection_fidelity(matrices[key], basis[: red["transfer_rank"]]) for key in test_keys}

    same_alignment = []
    different_alignment = []
    for left, right in combinations(matrices, 2):
        if metadata[left]["pair"] == metadata[right]["pair"]:
            continue
        alignment = subspace_alignment(path_bases[left], path_bases[right])
        (same_alignment if metadata[left]["family"] == metadata[right]["family"] else different_alignment).append(alignment)

    family32_values = [row["family32"] for row in member_rows.values()]
    balanced32_values = [row["global_balanced32"] for row in member_rows.values()]
    unbalanced32_values = [row["global_unbalanced32"] for row in member_rows.values()]
    leave_values = [value for pair in leave_pair_out.values() for value in pair.values()]
    hypotheses = {
        "B1_every_individual_semantic_path_is_low_rank": {"supported": all(row["individual_rank90"] <= 16 for row in member_rows.values()), "maximum_rank90": max(row["individual_rank90"] for row in member_rows.values())},
        "B2_each_balanced_semantic_family_is_compact": {"supported": all(row["summary"]["rank90"] <= 32 for row in family_results.values()), "ranks": {family: row["summary"]["rank90"] for family, row in family_results.items()}},
        "B3_a_balanced_global_semantic_subspace_is_moderately_compact": {"supported": bool(global_balanced["rank90"] <= 64), "rank90": global_balanced["rank90"]},
        "B4_family_rank32_is_robust_per_context": {"supported": bool(min(family32_values) >= 0.75), "minimum": min(family32_values)},
        "B5_leave_one_pair_out_family_transfer_is_useful": {"supported": bool(min(leave_values) >= 0.50), "minimum": min(leave_values), "mean": float(np.mean(leave_values))},
        "B6_family_bases_beat_the_global_basis_at_fixed_rank": {"supported": bool(np.mean(family32_values) - np.mean(balanced32_values) >= 0.10), "advantage": float(np.mean(family32_values) - np.mean(balanced32_values)), "family_mean": float(np.mean(family32_values)), "global_mean": float(np.mean(balanced32_values))},
        "B7_same_family_path_subspaces_are_more_aligned": {"supported": bool(np.mean(same_alignment) - np.mean(different_alignment) >= 0.05), "advantage": float(np.mean(same_alignment) - np.mean(different_alignment)), "same": float(np.mean(same_alignment)), "different": float(np.mean(different_alignment))},
        "B8_unbalanced_global_rank_hides_low_energy_paths": {"supported": bool(min(balanced32_values) - min(unbalanced32_values) >= 0.10), "worst_balanced": min(balanced32_values), "worst_unbalanced": min(unbalanced32_values), "gap": min(balanced32_values) - min(unbalanced32_values)},
        "B9_balanced_global_rank64_is_robust_per_context": {"supported": bool(min(row["global_balanced64"] for row in member_rows.values()) >= 0.75), "minimum": min(row["global_balanced64"] for row in member_rows.values())},
        "B10_finite_mlp_changes_are_captured_by_family_bases": {"supported": bool(min(row["family_finite32"] for row in member_rows.values()) >= 0.90), "minimum": min(row["family_finite32"] for row in member_rows.values())},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "individual": individual, "family_results": {family: row["summary"] for family, row in family_results.items()}, "global_balanced": global_balanced, "global_unbalanced": global_unbalanced, "members": member_rows, "leave_pair_out": leave_pair_out, "alignment": {"same_family": float(np.mean(same_alignment)), "different_family": float(np.mean(different_alignment))}}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    axes[0].bar(np.arange(len(families)), [family_results[family]["summary"]["rank90"] for family in families])
    axes[0].set_xticks(np.arange(len(families)))
    axes[0].set_xticklabels(families, rotation=30, ha="right")
    axes[0].set_title("Balanced family rank to 90%")
    axes[1].hist(family32_values, bins=12, alpha=0.7, label="family32")
    axes[1].hist(balanced32_values, bins=12, alpha=0.5, label="global32")
    axes[1].hist(unbalanced32_values, bins=12, alpha=0.4, label="unbalanced32")
    axes[1].set_title("Per-context reconstruction")
    axes[2].bar(["same family", "different family"], [np.mean(same_alignment), np.mean(different_alignment)])
    axes[2].set_title("Rank-8 path-subspace alignment")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Balanced semantic path subspaces for the layer-0 bilinear MLP",
        "",
        "Every one of the 38 semantic edit contexts contributes equal Frobenius energy before fitting family and global output subspaces. Per-context, leave-one-lexical-pair-out, finite-delta, and canonical-alignment tests prevent a high-energy path from defining a misleadingly small aggregate rank.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Balanced semantic subspaces](bilin12_semantic_layer0_balanced_subspaces.png)",
        "",
        "| Family | Balanced rank90/rank99 | Worst member family-rank32 | Mean held-out-pair rank32 |",
        "|---|---:|---:|---:|",
    ]
    for family in families:
        keys = [key for key, row in metadata.items() if row["family"] == family]
        pairs = {metadata[key]["pair"] for key in keys}
        held = [value for pair in pairs for value in leave_pair_out[pair].values()]
        lines.append(f"| {family} | {family_results[family]['summary']['rank90']}/{family_results[family]['summary']['rank99']} | {min(member_rows[key]['family32'] for key in keys):.3f} | {np.mean(held):.3f} |")
    lines += [
        "",
        f"Balanced global rank-90%: **{global_balanced['rank90']}**; unbalanced rank-90%: **{global_unbalanced['rank90']}**. Worst per-context global rank-32 fidelity is **{min(balanced32_values):.3f}** balanced versus **{min(unbalanced32_values):.3f}** unbalanced. Same-family versus different-family rank-8 alignment is **{np.mean(same_alignment):.3f}** versus **{np.mean(different_alignment):.3f}**.",
        "",
        "Equal path weighting asks for a reusable coordinate system rather than a variance-optimal summary of the most energetic edits. Leave-pair-out transfer is the stronger semantic test: a compact in-sample union can still be a lookup table over observed trajectories.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
