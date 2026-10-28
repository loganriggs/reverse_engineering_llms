#!/usr/bin/env python3
"""Canonicalize the exact quadratic router into the pooled write backbone."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_pooled_router_tensor_canonicalization.json")
OUTPUT = TENSOR_ROOT / "bilin12_pooled_router_tensor_canonicalization.json"
REPORT = TENSOR_ROOT / "BILIN12_POOLED_ROUTER_TENSOR_CANONICALIZATION.md"
FIGURE = TENSOR_ROOT / "bilin12_pooled_router_tensor_canonicalization.png"


def coefficient_metrics(target, predicted):
    centered = target - target.mean(0)
    r2 = 1 - (target - predicted).square().sum() / centered.square().sum().clamp_min(1e-30)
    cosine = F.cosine_similarity(target, predicted, dim=-1)
    return {"r2": float(r2), "median_cosine": float(cosine.median())}


def coefficients(forms, basis, base, edit):
    base_coordinates = base @ basis
    edit_coordinates = edit @ basis
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis)
    cross = 2 * torch.einsum("bi,kij,bj->bk", base_coordinates, restricted, edit_coordinates)
    square = torch.einsum("bi,kij,bj->bk", edit_coordinates, restricted, edit_coordinates)
    return cross + square


def first_passing(results, corpus, oracle, ranks):
    for rank in ranks:
        row = results[corpus]["frontier"][str(rank)]["weight"]
        if row["fidelity"] >= .70 and row["fidelity"] >= oracle["fidelity"] - .15:
            return rank
    return None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"]})
    block = model.blocks[11]
    left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    # Each output coefficient is x^T S_k x. Only the symmetric part is observable.
    raw_forms = torch.einsum("ja,jb,jk->kab", left, right, channel_to_output)
    forms = (raw_forms + raw_forms.transpose(-1, -2)) / 2
    gram = torch.einsum("kab,kcb->ac", forms, forms)
    _, weight_vectors = torch.linalg.eigh(gram)
    weight_vectors = weight_vectors.flip(-1).contiguous()
    train_panel = paired_panel(model, config["discovery"], 11, target, config["batch_size"])
    dimension = model.config.n_embd
    train_base = F.rms_norm(train_panel[0], (dimension,)).float()
    train_changed = F.rms_norm(train_panel[3], (dimension,)).float()
    _, _, state_vh = torch.linalg.svd(torch.cat((train_base, train_changed)), full_matrices=False)
    state_vectors = state_vh.T.contiguous()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_vectors = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator,
                                                  device="cuda")).Q
    results = {}; maximum_relative_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, 11, target, config["batch_size"])
        base = F.rms_norm(panel[0], (dimension,)).float()
        changed = F.rms_norm(panel[3], (dimension,)).float(); edit = changed - base
        target_coefficients = ((panel[4] - panel[1]).float() @ write_basis)
        exact = coefficients(forms, torch.eye(dimension, device="cuda"), base, edit)
        relative = ((exact - target_coefficients).norm(dim=-1)
                    / target_coefficients.norm(dim=-1).clamp_min(1e-30))
        maximum_relative_error = max(maximum_relative_error, float(relative.max()))
        oracle = causal(model, panel, target_coefficients @ write_basis.T)
        frontier = {}
        for rank in config["input_ranks"]:
            rows = {}
            for name, vectors in (("weight", weight_vectors), ("state_pca", state_vectors),
                                  ("random", random_vectors)):
                predicted_coefficients = coefficients(forms, vectors[:, :rank], base, edit)
                metric = causal(model, panel, predicted_coefficients @ write_basis.T)
                rows[name] = {**metric, **coefficient_metrics(target_coefficients,
                                                               predicted_coefficients)}
            frontier[str(rank)] = rows
        results[corpus] = {"examples": len(base), "oracle": oracle, "frontier": frontier}
    gate = lambda row, oracle: row["fidelity"] >= .70 and row["fidelity"] >= oracle["fidelity"] - .15
    q0 = maximum_relative_error <= 1e-5
    q1 = all(gate(row["frontier"]["64"]["weight"], row["oracle"]) for row in results.values())
    q2 = all(gate(row["frontier"]["128"]["weight"], row["oracle"]) for row in results.values())
    q3 = all(row["frontier"][str(rank)]["weight"]["fidelity"]
             >= row["frontier"][str(rank)]["random"]["fidelity"] + .20
             for row in results.values() for rank in (32, 64, 128))
    q4 = all(row["frontier"]["64"]["weight"]["fidelity"]
             >= row["frontier"]["64"]["state_pca"]["fidelity"] + .10
             for row in results.values())
    passing = {corpus: first_passing(results, corpus, row["oracle"], config["input_ranks"])
               for corpus, row in results.items()}
    q5 = any(all(gate(results[c]["frontier"][str(rank)]["weight"], results[c]["oracle"])
                 for c in results) for rank in config["input_ranks"] if rank <= 128)
    q6 = results["wikitext"]["frontier"]["128"]["weight"]["fidelity"] >= .707
    hypotheses = {
        "Q0_exact_tensor_map_recomposes_coefficients": {"supported": q0,
                                                          "maximum_relative_error": maximum_relative_error},
        "Q1_weight_rank64_is_sufficient": {"supported": q1},
        "Q2_weight_rank128_is_sufficient": {"supported": q2},
        "Q3_weight_subspace_beats_random": {"supported": q3},
        "Q4_weight_subspace_beats_state_pca": {"supported": q4},
        "Q5_one_compact_input_space_transfers_ood": {"supported": q5,
                                                       "minimum_passing_ranks": passing},
        "Q6_quadratic_map_repairs_linear_wikitext_failure": {"supported": q6},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for axis, corpus in zip(axes, results):
        row = results[corpus]
        for name in ("weight", "state_pca", "random"):
            axis.plot(config["input_ranks"], [row["frontier"][str(rank)][name]["fidelity"]
                                               for rank in config["input_ranks"]],
                      marker="o", label=name)
        axis.axhline(row["oracle"]["fidelity"], color="black", lw=.8, label="full oracle")
        axis.axhline(.70, color="black", ls="--", lw=.8)
        axis.set_xscale("log", base=2); axis.set_title(corpus); axis.set_xlabel("shared input rank")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Canonical quadratic router into the pooled write backbone", "",
             "The native Layer-11 bilinear weights and the frozen pooled rank-32 output basis define 32 exact symmetric quadratic forms. Their joint squared operator fixes a shared weight-canonical input basis. Truncating that basis tests whether the exact router has a compact, basis-invariant input description; FineWeb state PCA and Haar-random spaces are matched-rank controls.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Quadratic router](bilin12_pooled_router_tensor_canonicalization.png)", "",
              "| Corpus | Full oracle | Weight rank 32/64/128/256 | State-PCA rank 64 | Random rank 64 | Minimum passing weight rank |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        values = "/".join(f"{row['frontier'][str(rank)]['weight']['fidelity']:.3f}" for rank in (32,64,128,256))
        lines.append(f"| {corpus} | {row['oracle']['fidelity']:.3f} | {values} | "
                     f"{row['frontier']['64']['state_pca']['fidelity']:.3f} | "
                     f"{row['frontier']['64']['random']['fidelity']:.3f} | {passing[corpus] or 'none'} |")
    lines += ["", "The full quadratic map is an exact mechanistic explanation of this final-layer interface, not a learned probe. Truncation asks the separate description-length question. A high required rank would mean the computation is understood algebraically but is not small in this particular shared-input ontology."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
