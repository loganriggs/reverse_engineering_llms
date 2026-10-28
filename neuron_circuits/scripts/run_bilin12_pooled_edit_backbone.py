#!/usr/bin/env python3
"""Fit one shared late write basis and channel bank across token edits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_multi_operation_modules import activation_delta
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_pooled_edit_backbone.json")
OUTPUT = TENSOR_ROOT / "bilin12_pooled_edit_backbone.json"
REPORT = TENSOR_ROOT / "BILIN12_POOLED_EDIT_BACKBONE.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    multi = json.loads(Path(config["parent_config"]).read_text())
    cooperation = json.loads(Path(multi["parent_config"]).read_text())
    analytic = json.loads(Path(cooperation["parent_config"]).read_text())
    subspace = json.loads(Path(analytic["parent_config"]).read_text())
    state_config = json.loads(Path(subspace["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    tokens = {name: token_id(tokenizer, name) for name in multi["targets"]}
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    block = model.blocks[parent["layer"]]; down = block.mlp.Down.weight.float()
    max_rank = max(config["ranks"])
    discovery = {}; all_deltas = []
    for name, target in tokens.items():
        panel = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                             target, parent["batch_size"])
        delta = panel[4] - panel[1]; act = activation_delta(block, panel)
        _, _, vh = torch.linalg.svd(delta.float(), full_matrices=False)
        discovery[name] = {"basis": vh[:max_rank].T.contiguous(), "activation": act}
        all_deltas.append(delta)
    _, _, pooled_vh = torch.linalg.svd(torch.cat(all_deltas).float(), full_matrices=False)
    pooled_basis = pooled_vh[:max_rank].T.contiguous()
    channel_norm = (down.T @ pooled_basis).norm(dim=-1)
    pooled_importance = torch.stack([
        item["activation"].abs().mean(0) * channel_norm for item in discovery.values()
    ]).mean(0)
    pooled_order = pooled_importance.argsort(descending=True)
    target_orders = {}
    for name, item in discovery.items():
        target_norm = (down.T @ item["basis"]).norm(dim=-1)
        target_orders[name] = (item["activation"].abs().mean(0) * target_norm).argsort(descending=True)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_basis = torch.linalg.qr(torch.randn(model.config.n_embd, max_rank,
                                               generator=generator, device="cuda")).Q
    random_order = torch.randperm(down.shape[1], generator=generator, device="cuda")
    results = {}
    for name, target in tokens.items():
        rows = {}
        for corpus, spec in parent["evaluation"].items():
            print("evaluate", repr(name), corpus, flush=True)
            panel = paired_panel(model, spec, parent["layer"], target, parent["batch_size"])
            delta = panel[4] - panel[1]; act = activation_delta(block, panel)
            frontier = {}
            for rank in config["ranks"]:
                pooled = pooled_basis[:, :rank]; target_basis = discovery[name]["basis"][:, :rank]
                random = random_basis[:, :rank]
                projection = lambda basis: (delta @ basis) @ basis.T
                frontier[str(rank)] = {
                    "pooled": causal(model, panel, projection(pooled)),
                    "target": causal(model, panel, projection(target_basis)),
                    "random": causal(model, panel, projection(random)),
                }
            channels = {}
            primary_basis = pooled_basis[:, :config["primary_rank"]]
            for count in config["channel_counts"]:
                def selected(indices):
                    values = torch.zeros_like(act); values[:, indices] = act[:, indices]
                    write = block.mlp.Down(values)
                    return (write @ primary_basis) @ primary_basis.T
                channels[str(count)] = {
                    "pooled": causal(model, panel, selected(pooled_order[:count])),
                    "target": causal(model, panel, selected(target_orders[name][:count])),
                    "random": causal(model, panel, selected(random_order[:count])),
                }
            rows[corpus] = {"frontier": frontier, "channels": channels}
        results[name] = {"token_id": target, "corpora": rows}
    primary_rank = str(config["primary_rank"]); primary_channels = str(config["primary_channels"])
    all_rows = [(name, corpus, row) for name, operation in results.items()
                for corpus, row in operation["corpora"].items()]
    g0 = all(row["frontier"][primary_rank]["pooled"]["fidelity"]
             >= row["frontier"][primary_rank]["target"]["fidelity"] - .05
             for _, _, row in all_rows)
    g1 = all(row["frontier"]["64"]["pooled"]["fidelity"] >= .70
             for _, _, row in all_rows)
    g2 = all(row["channels"][primary_channels]["pooled"]["fidelity"]
             >= row["channels"][primary_channels]["target"]["fidelity"] - .05
             for _, _, row in all_rows)
    g3 = all(row["frontier"][primary_rank]["pooled"]["fidelity"]
             >= row["frontier"][primary_rank]["random"]["fidelity"] + .30
             for _, _, row in all_rows)
    g4 = all(row["channels"]["256"]["pooled"]["fidelity"] >= .60
             for _, _, row in all_rows)
    hypotheses = {
        "G0_pooled_rank32_matches_target_specific": {"supported": g0},
        "G1_pooled_rank64_is_broadly_sufficient": {"supported": g1},
        "G2_common_channel_bank_matches_target_specific": {"supported": g2},
        "G3_pooled_backbone_beats_random": {"supported": g3},
        "G4_one_shared_native_bank_is_useful": {"supported": g4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "tokens": tokens,
                                  "hypotheses": hypotheses, "results": results}, indent=2))
    lines = ["# Pooled late edit backbone", "",
             "One write basis and one native-channel ordering are learned from four FineWeb token replacement operations and frozen before evaluation.", "", "## Locked hypotheses", ""]
    for key, item in hypotheses.items():
        lines.append(f"- **{key}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Target | Corpus | Pooled/target/random rank-32 fidelity | Pooled rank-64 | Pooled/target/random top-128 fidelity | Pooled top-256 |",
              "|---|---|---:|---:|---:|---:|"]
    for name, operation in results.items():
        for corpus, row in operation["corpora"].items():
            rank = row["frontier"][primary_rank]; channels = row["channels"][primary_channels]
            lines.append(
                f"| `{name.encode('unicode_escape').decode()}` | {corpus} | "
                f"{rank['pooled']['fidelity']:.3f}/{rank['target']['fidelity']:.3f}/{rank['random']['fidelity']:.3f} | "
                f"{row['frontier']['64']['pooled']['fidelity']:.3f} | "
                f"{channels['pooled']['fidelity']:.3f}/{channels['target']['fidelity']:.3f}/{channels['random']['fidelity']:.3f} | "
                f"{row['channels']['256']['pooled']['fidelity']:.3f} |"
            )
    lines += ["", "A pooled success identifies a shared actuator space and implementation bank. It does not imply that the operations share upstream evidence or that their coefficient vectors have the same semantics."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
