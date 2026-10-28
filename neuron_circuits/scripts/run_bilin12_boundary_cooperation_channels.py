#!/usr/bin/env python3
"""Correct the shuffled bilinear control and measure native-channel sufficiency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_cooperation_channels.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_cooperation_channels.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_COOPERATION_CHANNELS.md"


def activation_terms(block, pre, changed_pre):
    dimension = pre.shape[-1]
    base = F.rms_norm(pre, (dimension,)); changed = F.rms_norm(changed_pre, (dimension,))
    edit = changed - base
    lb, rb = block.mlp.Left(base), block.mlp.Right(base)
    ld, rd = block.mlp.Left(edit), block.mlp.Right(edit)
    return base, lb * rd + ld * rb, ld * rd


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    analytic = json.loads(Path(config["parent_config"]).read_text())
    subspace = json.loads(Path(analytic["parent_config"]).read_text())
    state_config = json.loads(Path(subspace["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = token_id(tokenizer, "\n")
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    block = model.blocks[parent["layer"]]
    train = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                         newline, parent["batch_size"])
    train_delta = train[4] - train[1]
    _, _, vh = torch.linalg.svd(train_delta.float(), full_matrices=False)
    basis = vh[:analytic["rank"]].T.contiguous()
    down = block.mlp.Down.weight.float()
    channel_to_coeff = down.T @ basis
    _, train_cross, train_quadratic = activation_terms(block, train[0], train[3])
    train_activation = train_cross + train_quadratic
    channel_norm = channel_to_coeff.norm(dim=-1)
    importance = (train_activation.abs() * channel_norm).mean(0)
    fixed_order = importance.argsort(descending=True)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_order = torch.randperm(len(importance), generator=generator, device="cuda")
    project = lambda value: (value @ basis) @ basis.T
    results = {}
    for corpus, spec in parent["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, parent["layer"], newline, parent["batch_size"])
        base, cross_activation, quadratic_activation = activation_terms(block, panel[0], panel[3])
        activation = cross_activation + quadratic_activation
        cross_write = project(block.mlp.Down(cross_activation))
        quadratic_write = project(block.mlp.Down(quadratic_activation))
        full_write = cross_write + quadratic_write
        permutation = torch.randperm(len(base), generator=generator, device="cuda")
        dimension = base.shape[-1]
        edit = F.rms_norm(panel[3], (dimension,)) - base
        shuffled_cross_activation = (block.mlp.Left(base[permutation]) * block.mlp.Right(edit)
                                     + block.mlp.Left(edit) * block.mlp.Right(base[permutation]))
        shuffled_cross = project(block.mlp.Down(shuffled_cross_activation))
        shuffled_cross = shuffled_cross * (
            cross_write.norm(dim=-1, keepdim=True)
            / shuffled_cross.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        )
        shuffled_full = shuffled_cross + quadratic_write
        cosine = F.cosine_similarity(cross_write, quadratic_write, dim=-1)
        cancellation = (full_write.norm(dim=-1)
                        / (cross_write.norm(dim=-1) + quadratic_write.norm(dim=-1)).clamp_min(1e-30))
        frontier = {}
        contribution = activation.abs() * channel_norm
        for count in config["channel_counts"]:
            fixed_indices = fixed_order[:count]; random_indices = random_order[:count]
            fixed_activation = torch.zeros_like(activation)
            random_activation = torch.zeros_like(activation)
            fixed_activation[:, fixed_indices] = activation[:, fixed_indices]
            random_activation[:, random_indices] = activation[:, random_indices]
            oracle_indices = contribution.topk(count, dim=-1).indices
            oracle_activation = torch.zeros_like(activation)
            oracle_activation.scatter_(1, oracle_indices,
                                       activation.gather(1, oracle_indices))
            frontier[str(count)] = {
                "fixed": causal(model, panel, project(block.mlp.Down(fixed_activation))),
                "random": causal(model, panel, project(block.mlp.Down(random_activation))),
                "oracle": causal(model, panel, project(block.mlp.Down(oracle_activation))),
            }
        results[corpus] = {
            "examples": len(base), "full": causal(model, panel, full_write),
            "cross": causal(model, panel, cross_write),
            "quadratic": causal(model, panel, quadratic_write),
            "shuffled_full": causal(model, panel, shuffled_full),
            "cross_quadratic_cosine_median": float(cosine.median()),
            "cancellation_ratio_median": float(cancellation.median()),
            "frontier": frontier,
        }
    c0 = all(row["shuffled_full"]["fidelity"] <= row["full"]["fidelity"] - .20
             for row in results.values())
    c1 = all(row["cross_quadratic_cosine_median"] <= -.50 for row in results.values())
    c2 = all(row["cancellation_ratio_median"] <= .50 for row in results.values())
    p0 = all(row["frontier"]["128"]["fixed"]["fidelity"]
             >= row["full"]["fidelity"] - .10 for row in results.values())
    p1 = all(row["frontier"]["32"]["fixed"]["fidelity"] >= .50
             for row in results.values())
    p2 = all(row["frontier"]["64"]["fixed"]["fidelity"]
             >= row["frontier"]["64"]["random"]["fidelity"] + .30
             for row in results.values())
    p3 = all(row["frontier"]["32"]["oracle"]["fidelity"]
             >= row["frontier"]["32"]["fixed"]["fidelity"] + .15
             for row in results.values())
    hypotheses = {
        "C0_correct_base_pairing_is_necessary_in_full_sum": {"supported": c0},
        "C1_cross_and_quadratic_writes_cancel": {"supported": c1},
        "C2_cancellation_is_large": {"supported": c2},
        "P0_fixed_128_channels_nearly_recover_module": {"supported": p0},
        "P1_fixed_32_channels_are_useful": {"supported": p1},
        "P2_weighted_native_selection_beats_random": {"supported": p2},
        "P3_contextual_channel_selection_matters": {"supported": p3},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))
    lines = ["# Cooperative bilinear terms and native-channel frontier", "",
             "The shuffled-base correction evaluates shuffled cross terms together with the intact edit-squared term. Native channel sets are ranked on disjoint FineWeb discovery examples and compared with random and per-example oracle sets.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Full/shuffled-full fidelity | Cross–quadratic cosine | Cancellation ratio | Fixed/random/oracle 32 fidelity | Fixed 128 fidelity |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        f32 = row["frontier"]["32"]
        lines.append(
            f"| {corpus} | {row['full']['fidelity']:.3f}/{row['shuffled_full']['fidelity']:.3f} | "
            f"{row['cross_quadratic_cosine_median']:.3f} | {row['cancellation_ratio_median']:.3f} | "
            f"{f32['fixed']['fidelity']:.3f}/{f32['random']['fidelity']:.3f}/{f32['oracle']['fidelity']:.3f} | "
            f"{row['frontier']['128']['fixed']['fidelity']:.3f} |"
        )
    lines += ["", "The fixed sets test a reusable native module. The oracle sets test context-dependent multiplexing at the same channel budget. Neither metric by itself assigns a human meaning to individual channels."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
