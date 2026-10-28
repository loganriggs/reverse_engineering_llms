#!/usr/bin/env python3
"""Ask whether the compact natural-state router separates into scalar axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_joint_diagonalization.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_joint_diagonalization.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JOINT_DIAGONALIZATION.md"
FIGURE = TENSOR_ROOT / "bilin12_router_joint_diagonalization.png"


def evaluate_forms(forms, base, edit):
    return (2 * torch.einsum("bi,kij,bj->bk", base, forms, edit)
            + torch.einsum("bi,kij,bj->bk", edit, forms, edit))


def masked_forms(forms, pairs, count):
    result = torch.diag_embed(forms.diagonal(dim1=-2, dim2=-1))
    for index in pairs[:count]:
        i, j = index
        result[:, i, j] = forms[:, i, j]
        result[:, j, i] = forms[:, j, i]
    return result


def discovery_pair_order(forms, base, edit):
    dimension = forms.shape[-1]; rows = []
    for i in range(dimension):
        for j in range(i + 1, dimension):
            scalar = 2 * (base[:, i] * edit[:, j] + base[:, j] * edit[:, i]
                          + edit[:, i] * edit[:, j])
            contribution = scalar[:, None] * forms[:, i, j][None]
            rows.append((float(contribution.square().sum(-1).mean()), (i, j)))
    return [pair for _, pair in sorted(rows, reverse=True)]


def weight_pair_order(forms):
    rows = []
    for i in range(forms.shape[-1]):
        for j in range(i + 1, forms.shape[-1]):
            rows.append((float(forms[:, i, j].square().sum()), (i, j)))
    return [pair for _, pair in sorted(rows, reverse=True)]


def row_best_overlap(reference, candidate):
    values = (reference.T @ candidate).abs()
    return float(values.max(-1).values.mean()), int(values.argmax(-1).unique().numel())


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural_states, _ = collect_states_writes(model, 11, config["natural"],
                                               config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural_states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, training = train_rotation(restricted, config)
    rotation = rotation.detach(); rotated = rotation.T[None] @ restricted @ rotation[None]
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_rotation = torch.linalg.qr(torch.randn(config["input_rank"], config["input_rank"],
                                                   generator=generator, device="cuda")).Q
    random_forms = random_rotation.T[None] @ restricted @ random_rotation[None]
    discovery = paired_panel(model, config["discovery"], 11, target, config["batch_size"])
    discovery_base = F.rms_norm(discovery[0], (dimension,)).float() @ natural_basis @ rotation
    discovery_changed = F.rms_norm(discovery[3], (dimension,)).float() @ natural_basis @ rotation
    discovery_order = discovery_pair_order(rotated, discovery_base, discovery_changed - discovery_base)
    weight_order = weight_pair_order(rotated)
    random_order = list(weight_order)
    rng = np.random.default_rng(config["seed"]); rng.shuffle(random_order)

    starts = [training]; overlaps = []
    torch.manual_seed(config["seed"])
    for _ in range(3):
        initial = .01 * torch.randn(config["input_rank"], config["input_rank"], device="cuda",
                                     dtype=torch.float64)
        candidate, row = train_rotation(restricted, config, initial)
        match, unique = row_best_overlap(rotation, candidate.detach())
        row["mean_best_axis_overlap"] = match; row["unique_best_axes"] = unique
        starts.append(row); overlaps.append(match)
    losses = np.asarray([row["final_off_diagonal_fraction"] for row in starts])
    stability = {"loss_mean": float(losses.mean()), "loss_cv": float(losses.std() / losses.mean()),
                 "mean_best_axis_overlap": float(np.mean(overlaps)), "starts": starts}

    results = {}
    with torch.inference_mode():
        for corpus, spec in config["evaluation"].items():
            print("evaluate", corpus, flush=True)
            panel = paired_panel(model, spec, 11, target, config["batch_size"])
            base_full = F.rms_norm(panel[0], (dimension,)).float()
            changed_full = F.rms_norm(panel[3], (dimension,)).float()
            base = base_full @ natural_basis @ rotation; changed = changed_full @ natural_basis @ rotation
            edit = changed - base
            pca_base = base_full @ natural_basis; pca_edit = changed_full @ natural_basis - pca_base
            random_base = pca_base @ random_rotation; random_edit = pca_edit @ random_rotation
            target_coefficients = (panel[4] - panel[1]).float() @ write_basis
            methods = {
                "full": evaluate_forms(rotated, base, edit),
                "diagonal": evaluate_forms(masked_forms(rotated, [], 0), base, edit),
                "pca_diagonal": evaluate_forms(masked_forms(restricted, [], 0), pca_base, pca_edit),
                "random_diagonal": evaluate_forms(masked_forms(random_forms, [], 0), random_base, random_edit),
            }
            method_metrics = {name: causal(model, panel, value @ write_basis.T)
                              for name, value in methods.items()}
            frontiers = {}
            for count in config["off_diagonal_budgets"]:
                frontiers[str(count)] = {}
                for name, order in (("discovery", discovery_order), ("weight", weight_order),
                                    ("random", random_order)):
                    value = evaluate_forms(masked_forms(rotated, order, count), base, edit)
                    frontiers[str(count)][name] = causal(model, panel, value @ write_basis.T)
            results[corpus] = {"oracle": causal(model, panel, target_coefficients @ write_basis.T),
                               "methods": method_metrics, "pair_frontiers": frontiers}
    j0 = training["relative_reduction"] >= .70
    j1 = all(row["methods"]["diagonal"]["fidelity"] >= row["methods"]["full"]["fidelity"] - .10
             for row in results.values())
    j2 = all(row["methods"]["diagonal"]["fidelity"]
             >= row["methods"]["pca_diagonal"]["fidelity"] + .10 for row in results.values())
    j3 = all(row["methods"]["diagonal"]["fidelity"]
             >= row["methods"]["random_diagonal"]["fidelity"] + .20 for row in results.values())
    j4 = stability["loss_cv"] <= .05 and stability["mean_best_axis_overlap"] >= .90
    j5 = any(row["methods"]["full"]["fidelity"] >= row["methods"]["diagonal"]["fidelity"] + .10
             for row in results.values())
    j6 = all(row["pair_frontiers"]["64"]["discovery"]["fidelity"]
             >= row["methods"]["full"]["fidelity"] - .05 for row in results.values())
    j7 = all(any(row["pair_frontiers"][str(count)]["discovery"]["fidelity"]
                     >= row["pair_frontiers"][str(count)]["weight"]["fidelity"] + .05
                     for count in (32,64)) for row in results.values())
    hypotheses = {
        "J0_forms_are_approximately_jointly_diagonalizable": {"supported": j0, "training": training},
        "J1_diagonal_program_is_sufficient": {"supported": j1},
        "J2_rotation_improves_diagonal_program": {"supported": j2},
        "J3_diagonal_program_beats_random_rotation": {"supported": j3},
        "J4_canonical_axes_are_multistart_stable": {"supported": j4, "stability": stability},
        "J5_cross_terms_are_causally_necessary": {"supported": j5},
        "J6_sixtyfour_cross_pairs_are_sufficient": {"supported": j6},
        "J7_discovery_pair_ranking_beats_weight_only": {"supported": j7},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "training": training, "stability": stability,
                                  "results": results}, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for axis, (corpus, row) in zip(axes, results.items()):
        for name in ("discovery", "weight", "random"):
            axis.plot(config["off_diagonal_budgets"],
                      [row["pair_frontiers"][str(count)][name]["fidelity"]
                       for count in config["off_diagonal_budgets"]], marker="o", label=name)
        axis.axhline(row["methods"]["full"]["fidelity"], color="black", lw=.8, label="full rank-32")
        axis.set_title(corpus); axis.set_xlabel("retained off-diagonal pairs"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Joint diagonalization of the compact quadratic router", "",
             "The exact pooled-output forms are restricted to one ordinary-state rank-32 basis, then rotated using weights only to minimize joint off-diagonal energy. Diagonal-only execution asks whether the computation becomes independent scalar coordinates; frozen interaction-pair frontiers test how many relational terms are still needed.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Interaction frontier](bilin12_router_joint_diagonalization.png)", "",
              f"Off-diagonal energy: `{training['initial_off_diagonal_fraction']:.3f}` → `{training['final_off_diagonal_fraction']:.3f}`; multistart loss CV `{stability['loss_cv']:.3f}`, axis overlap `{stability['mean_best_axis_overlap']:.3f}`.", "",
              "| Corpus | Full / optimized diagonal / PCA diagonal / random diagonal | Discovery pairs 16/32/64/128 | Weight pairs 64 |",
              "|---|---:|---:|---:|"]
    for corpus, row in results.items():
        methods = row["methods"]
        method_text = "/".join(f"{methods[name]['fidelity']:.3f}" for name in
                               ("full", "diagonal", "pca_diagonal", "random_diagonal"))
        discovery_text = "/".join(f"{row['pair_frontiers'][str(count)]['discovery']['fidelity']:.3f}"
                                  for count in (16,32,64,128))
        lines.append(f"| {corpus} | {method_text} | {discovery_text} | "
                     f"{row['pair_frontiers']['64']['weight']['fidelity']:.3f} |")
    lines += ["", "Joint diagonalization can reveal a neuron-like scalar code only if diagonal execution preserves the finite behavioral effect. Residual off-diagonal terms are explicit pairwise verbs in the compressed coordinate system—not evidence that either coordinate has two unrelated meanings. Axis multistart stability and pair-ranking controls prevent a visually sparse but non-identifiable rotation from being overinterpreted."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
