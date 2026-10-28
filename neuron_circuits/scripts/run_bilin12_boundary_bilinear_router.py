#!/usr/bin/env python3
"""Decompose boundary-write routing into exact bilinear context cross terms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_output_interface import effect_similarity, logits_from_hidden, token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_bilinear_router.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_bilinear_router.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_BILINEAR_ROUTER.md"


@torch.inference_mode()
def causal(model, panel, delta):
    pre, write, base_logits, _, changed_write = panel
    oracle = logits_from_hidden(model, pre + changed_write) - base_logits
    predicted = logits_from_hidden(model, pre + write + delta) - base_logits
    cosine, fidelity = effect_similarity(base_logits, oracle, predicted)
    return {"cosine": cosine, "fidelity": fidelity}


def effective_support(values):
    values = values.abs()
    return values.sum(-1).square() / values.square().sum(-1).clamp_min(1e-30)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    subspace = json.loads(Path(config["parent_config"]).read_text())
    state_config = json.loads(Path(subspace["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = token_id(tokenizer, "\n")
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    block = model.blocks[parent["layer"]]
    train_panel = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                               newline, parent["batch_size"])
    train_delta = (train_panel[4] - train_panel[1]).float()
    _, _, vh = torch.linalg.svd(train_delta, full_matrices=False)
    basis = vh[:config["rank"]].T.contiguous()
    down = block.mlp.Down.weight.float()
    channel_to_coeff = down.T @ basis
    results = {}; maximum_error = 0.0
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    for corpus, spec in parent["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, parent["layer"], newline, parent["batch_size"])
        pre, write, _, changed_pre, changed_write = panel
        dimension = pre.shape[-1]
        base_state = F.rms_norm(pre, (dimension,))
        changed_state = F.rms_norm(changed_pre, (dimension,))
        edit = changed_state - base_state
        left_base, right_base = block.mlp.Left(base_state), block.mlp.Right(base_state)
        left_edit, right_edit = block.mlp.Left(edit), block.mlp.Right(edit)
        cross_activation = left_base * right_edit + left_edit * right_base
        quadratic_activation = left_edit * right_edit
        full_activation = cross_activation + quadratic_activation
        cross_write = block.mlp.Down(cross_activation)
        quadratic_write = block.mlp.Down(quadratic_activation)
        full_write = block.mlp.Down(full_activation)
        exact = changed_write - write
        relative = (full_write - exact).norm(dim=-1) / exact.norm(dim=-1).clamp_min(1e-30)
        maximum_error = max(maximum_error, float(relative.max()))
        project = lambda value: (value @ basis) @ basis.T
        projected_full = project(full_write)
        projected_cross = project(cross_write)
        projected_quadratic = project(quadratic_write)
        permutation = torch.randperm(len(base_state), generator=generator, device="cuda")
        shuffled_cross_activation = (left_base[permutation] * right_edit
                                     + left_edit * right_base[permutation])
        shuffled_cross = project(block.mlp.Down(shuffled_cross_activation))
        shuffled_cross = shuffled_cross * (
            projected_cross.norm(dim=-1, keepdim=True)
            / shuffled_cross.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        )
        contributions = full_activation[:, :, None] * channel_to_coeff[None]
        magnitude = contributions.norm(dim=-1)
        support = effective_support(magnitude)
        top32_fraction = magnitude.topk(32, dim=-1).values.sum(-1) / magnitude.sum(-1).clamp_min(1e-30)
        results[corpus] = {
            "examples": len(pre), "full": causal(model, panel, projected_full),
            "cross": causal(model, panel, projected_cross),
            "quadratic": causal(model, panel, projected_quadratic),
            "shuffled_cross": causal(model, panel, shuffled_cross),
            "effective_channel_support_median": float(support.median()),
            "top32_absolute_contribution_fraction_median": float(top32_fraction.median()),
            "identity_relative_error_max": float(relative.max()),
        }
    b0 = maximum_error <= 1e-5
    b1 = all(row["full"]["fidelity"] >= .80 for row in results.values())
    b2 = all(row["cross"]["fidelity"] >= row["full"]["fidelity"] - .10
             for row in results.values())
    b3 = all(row["quadratic"]["fidelity"] <= row["full"]["fidelity"] - .20
             for row in results.values())
    b4 = all(row["shuffled_cross"]["fidelity"] <= row["cross"]["fidelity"] - .20
             for row in results.values())
    b5 = all(row["effective_channel_support_median"] >= 256 for row in results.values())
    hypotheses = {
        "B0_bilinear_delta_identity_is_exact": {"supported": b0,
                                                  "maximum_error": maximum_error},
        "B1_rank32_analytic_module_is_sufficient": {"supported": b1},
        "B2_context_cross_terms_dominate": {"supported": b2},
        "B3_edit_squared_term_is_insufficient": {"supported": b3},
        "B4_correct_base_state_is_causally_necessary": {"supported": b4},
        "B5_compact_output_module_uses_distributed_native_channels": {"supported": b5},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))
    lines = ["# Analytic bilinear router for the compact boundary module", "",
             "The exact Layer-11 token-to-newline write change is split into context cross terms and an edit-squared term, then projected into the frozen FineWeb rank-32 write subspace. Shuffled base states are norm matched.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum analytic write-identity relative error: `{maximum_error:.2e}`.", "",
              "| Corpus | Full/cross/quadratic fidelity | Shuffled-cross fidelity | Effective native support | Top-32 contribution fraction |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {row['full']['fidelity']:.3f}/{row['cross']['fidelity']:.3f}/{row['quadratic']['fidelity']:.3f} | "
            f"{row['shuffled_cross']['fidelity']:.3f} | {row['effective_channel_support_median']:.1f} | "
            f"{row['top32_absolute_contribution_fraction_median']:.3f} |"
        )
    lines += ["", "Effective support is computed from absolute norms of each native channel's contribution to the 32 subspace coefficients. It measures implementation breadth, not semantic polysemanticity."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
