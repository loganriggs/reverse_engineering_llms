#!/usr/bin/env python3
"""Canonicalize diffuse native MLP-channel paths into output subspaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_his_her_layer0_mlp_path import layer0_components, path_row
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer0_mlp_path_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer0_mlp_path_subspace.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER0_MLP_PATH_SUBSPACE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer0_mlp_path_subspace.png"


def svd_summary(matrix, ranks):
    _, singular, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    energy = singular.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    rank90 = int(torch.searchsorted(cumulative, torch.tensor(0.90, device=cumulative.device)).item() + 1)
    rank99 = int(torch.searchsorted(cumulative, torch.tensor(0.99, device=cumulative.device)).item() + 1)
    fidelity = {str(rank): float(cumulative[min(rank, len(cumulative)) - 1]) for rank in ranks}
    return {"rank90": rank90, "rank99": rank99, "fidelity": fidelity}, right


def projection_fidelity(matrix, basis):
    prediction = (matrix @ basis.T) @ basis
    return float(1 - (matrix - prediction).square().sum() / matrix.square().sum().clamp_min(1e-30))


def vector_projection_fidelity(vector, basis):
    prediction = (vector @ basis.T) @ basis
    return float(1 - (vector - prediction).square().sum() / vector.square().sum().clamp_min(1e-30))


def subspace_alignment(left, right):
    singular = torch.linalg.svdvals(left @ right.T)
    return float(singular.square().mean())


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
    selected = [record for record in prior["records"] if record["family"] == red["family"]]
    locations = np.linspace(0, 1, red["dense_points"])
    weights = torch.as_tensor(np.ones(red["dense_points"]) / (red["dense_points"] - 1), device="cuda", dtype=torch.float32)
    weights[[0, -1]] *= 0.5
    records = []
    matrices = {}
    bases = {}

    for prior_record in selected:
        corpus = prior_record["corpus"]
        row_index, start = prior_record["row"], prior_record["start"]
        _, target_token = (int(value) for value in prior_record["pair"].split("->"))
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
        output_matrix = torch.cat([row["derivative"]["mlp"] for row in path], dim=0)
        coefficient_matrix = torch.cat([row["derivative"]["product"] for row in path], dim=0)
        output_summary, output_basis = svd_summary(output_matrix, red["ranks"])
        coefficient_summary, coefficient_basis = svd_summary(coefficient_matrix, red["ranks"])
        finite_mlp = changed_components[7].float() - base_components[7].float()
        finite_fidelity = {str(rank): vector_projection_fidelity(finite_mlp, output_basis[:rank]) for rank in red["ranks"]}
        key = (corpus, prior_record["pair"])
        matrices[key] = output_matrix
        bases[key] = output_basis
        records.append({
            "corpus": corpus,
            "pair": prior_record["pair"],
            "output": output_summary,
            "coefficient": coefficient_summary,
            "finite_mlp_fidelity": finite_fidelity,
        })
        print(corpus, prior_record["pair"], flush=True)

    pairs = sorted({record["pair"] for record in records})
    joint = {}
    for pair in pairs:
        matrix = torch.cat([matrices[(corpus, pair)] for corpus in config["evaluation"]], dim=0)
        summary, basis = svd_summary(matrix, red["ranks"])
        joint[pair] = summary
        bases[("joint", pair)] = basis
    all_matrix = torch.cat([matrices[key] for key in sorted(matrices)], dim=0)
    all_gender, all_basis = svd_summary(all_matrix, red["ranks"])

    anomaly = red["anomalous_pair"]
    rank = 16
    cross_corpus = {}
    corpora = list(config["evaluation"])
    for train in corpora:
        test = next(corpus for corpus in corpora if corpus != train)
        cross_corpus[f"{train}_to_{test}"] = projection_fidelity(matrices[(test, anomaly)], bases[(train, anomaly)][:rank])
    anomaly_alignment = subspace_alignment(bases[(corpora[0], anomaly)][:rank], bases[(corpora[1], anomaly)][:rank])
    other_pairs = [pair for pair in pairs if pair != anomaly]
    cross_pair_values = []
    for train in corpora:
        basis = bases[(train, anomaly)][:rank]
        for test in corpora:
            for pair in other_pairs:
                cross_pair_values.append(projection_fidelity(matrices[(test, pair)], basis))
    within_transfer = float(np.mean(list(cross_corpus.values())))
    cross_pair_transfer = float(np.mean(cross_pair_values))
    anomaly_records = [record for record in records if record["pair"] == anomaly]
    control_records = [record for record in records if record["pair"] != anomaly]
    prior_channel64 = {"fineweb": 0.5253434181213379, "tinystories": 0.7462155222892761}
    hypotheses = {
        "S1_each_anomalous_mlp_path_is_low_dimensional": {"supported": all(record["output"]["rank90"] <= 16 for record in anomaly_records), "ranks": {record["corpus"]: record["output"]["rank90"] for record in anomaly_records}},
        "S2_native_channel_coefficients_follow_few_path_modes": {"supported": all(record["coefficient"]["rank90"] <= 16 for record in anomaly_records), "ranks": {record["corpus"]: record["coefficient"]["rank90"] for record in anomaly_records}},
        "S3_the_joint_anomalous_pair_has_a_compact_subspace": {"supported": bool(joint[anomaly]["rank90"] <= 32), "rank90": joint[anomaly]["rank90"]},
        "S4_the_path_subspace_transfers_across_corpora": {"supported": all(value >= 0.75 for value in cross_corpus.values()), "fidelities": cross_corpus},
        "S5_cross_corpus_subspaces_are_aligned": {"supported": bool(anomaly_alignment >= 0.25), "mean_squared_canonical_correlation": anomaly_alignment},
        "S6_the_subspace_is_edit_specific_not_a_generic_gender_basis": {"supported": bool(within_transfer - cross_pair_transfer >= 0.20), "within_pair": within_transfer, "other_pairs": cross_pair_transfer, "gap": within_transfer - cross_pair_transfer},
        "S7_sixteen_canonical_directions_beat_sixtyfour_native_channels": {"supported": all(record["output"]["fidelity"]["16"] > prior_channel64[record["corpus"]] for record in anomaly_records), "rank16": {record["corpus"]: record["output"]["fidelity"]["16"] for record in anomaly_records}, "native64": prior_channel64},
        "S8_the_finite_mlp_change_lies_in_the_path_subspace": {"supported": all(record["finite_mlp_fidelity"]["8"] >= 0.95 for record in anomaly_records), "fidelities": {record["corpus"]: record["finite_mlp_fidelity"]["8"] for record in anomaly_records}},
        "S9_sharpness_does_not_require_higher_path_rank": {"supported": bool(np.median([record["output"]["rank90"] for record in anomaly_records]) <= np.median([record["output"]["rank90"] for record in control_records])), "anomaly_median": float(np.median([record["output"]["rank90"] for record in anomaly_records])), "control_median": float(np.median([record["output"]["rank90"] for record in control_records]))},
        "S10_gender_paths_share_a_moderately_compact_joint_subspace": {"supported": bool(all_gender["rank90"] <= 64), "rank90": all_gender["rank90"]},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "records": records, "joint_by_pair": joint, "all_gender": all_gender, "cross_corpus": cross_corpus, "cross_pair_transfer": cross_pair_transfer, "anomaly_alignment": anomaly_alignment}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for record in anomaly_records:
        axes[0].plot(red["ranks"], [record["output"]["fidelity"][str(rank)] for rank in red["ranks"]], marker="o", label=f"output {record['corpus']}")
        axes[0].plot(red["ranks"], [record["coefficient"]["fidelity"][str(rank)] for rank in red["ranks"]], ls="--", marker="o", label=f"coeff {record['corpus']}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("his→her path singular spectrum")
    axes[0].set_ylabel("trajectory energy fidelity")
    axes[1].bar(np.arange(len(pairs)), [joint[pair]["rank90"] for pair in pairs])
    axes[1].set_xticks(np.arange(len(pairs)))
    axes[1].set_xticklabels(pairs, rotation=30, ha="right")
    axes[1].set_title("Joint two-corpus rank to 90%")
    axes[2].bar(["his→her cross-corpus", "his→her to other gender"], [within_transfer, cross_pair_transfer])
    axes[2].set_title("Rank-16 subspace transfer")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Canonical path subspaces for diffuse layer-0 MLP channels",
        "",
        "Hundreds of native bilinear channels were required to reproduce the sharp local MLP derivative. This audit stacks the derivative along each 129-point token path and computes uncentered singular subspaces in both residual-output space and native post-product coefficient space. Cross-corpus and cross-edit transfer distinguish per-path compression from a reusable semantic basis.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![MLP path subspaces](bilin12_layer0_mlp_path_subspace.png)",
        "",
        "| Corpus | Pair | Output rank90/rank99 | Coefficient rank90/rank99 | Output rank-16 fidelity | Finite-delta rank-8 fidelity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['corpus']} | {record['pair']} | {record['output']['rank90']}/{record['output']['rank99']} | {record['coefficient']['rank90']}/{record['coefficient']['rank99']} | {record['output']['fidelity']['16']:.3f} | {record['finite_mlp_fidelity']['8']:.3f} |")
    lines += [
        "",
        f"Joint his→her rank-90%: **{joint[anomaly]['rank90']}**. All-gender joint rank-90%: **{all_gender['rank90']}**. Rank-16 cross-corpus his→her transfer: **{within_transfer:.3f}**; transfer into other gender edits: **{cross_pair_transfer:.3f}**. Rank-16 canonical alignment: **{anomaly_alignment:.3f}**.",
        "",
        "A compact path subspace is a shorter coordinate system for one family of interventions. It becomes a reusable circuit description only to the extent that it transfers to new contexts and examples; per-path SVD alone is descriptive compression.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
