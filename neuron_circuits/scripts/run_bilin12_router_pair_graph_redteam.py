#!/usr/bin/env python3
"""Replicate sparse quadratic interaction graphs on disjoint discovery data."""

from __future__ import annotations

import json
from pathlib import Path

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
from run_bilin12_router_joint_diagonalization import (
    discovery_pair_order, evaluate_forms, masked_forms,
)
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_pair_graph_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_pair_graph_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_PAIR_GRAPH_REDTEAM.md"


def jaccard(left, right, count):
    a, b = set(left[:count]), set(right[:count])
    return len(a & b) / len(a | b)


def panel_coordinates(model, spec, target, batch_size, natural_basis, rotation):
    panel = paired_panel(model, spec, 11, target, batch_size)
    dimension = model.config.n_embd
    base = F.rms_norm(panel[0], (dimension,)).float() @ natural_basis @ rotation
    changed = F.rms_norm(panel[3], (dimension,)).float() @ natural_basis @ rotation
    return panel, base, changed - base


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, config["natural"], config["positions"],
                                       config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms,
                              natural_basis).detach()
    rotation, training = train_rotation(restricted, config); rotation = rotation.detach()
    forms = rotation.T[None] @ restricted @ rotation[None]
    with torch.inference_mode():
        discovery = {}
        for name in ("a", "b"):
            panel, base, edit = panel_coordinates(model, config[f"discovery_{name}"], target,
                                                   config["batch_size"], natural_basis, rotation)
            discovery[name] = discovery_pair_order(forms, base, edit)
        overlaps = {str(count): jaccard(discovery["a"], discovery["b"], count)
                    for count in config["budgets"]}
        all_pairs = [(i, j) for i in range(config["input_rank"])
                     for j in range(i + 1, config["input_rank"])]
        rng = np.random.default_rng(config["seed"])
        random_orders = []
        for _ in range(config["random_repetitions"]):
            order = list(all_pairs); rng.shuffle(order); random_orders.append(order)
        results = {}
        for corpus, spec in config["evaluation"].items():
            print("evaluate", corpus, flush=True)
            panel, base, edit = panel_coordinates(model, spec, target, config["batch_size"],
                                                   natural_basis, rotation)
            full = causal(model, panel, evaluate_forms(forms, base, edit) @ write_basis.T)
            budgets = {}
            for count in config["budgets"]:
                selected = {}
                for name in ("a", "b"):
                    value = evaluate_forms(masked_forms(forms, discovery[name], count), base, edit)
                    selected[name] = causal(model, panel, value @ write_basis.T)
                nulls = []
                for order in random_orders:
                    value = evaluate_forms(masked_forms(forms, order, count), base, edit)
                    nulls.append(causal(model, panel, value @ write_basis.T)["fidelity"])
                selected["random_mean"] = float(np.mean(nulls))
                selected["random_p95"] = float(np.quantile(nulls, .95))
                budgets[str(count)] = selected
            results[corpus] = {"full": full, "budgets": budgets}
    p0 = overlaps["32"] >= .50
    p1 = all(all(row["budgets"]["32"][name]["fidelity"] >= row["full"]["fidelity"] - .05
                 for name in ("a", "b")) for row in results.values())
    p2 = overlaps["32"] < .80 and all(abs(row["budgets"]["32"]["a"]["fidelity"]
                                          - row["budgets"]["32"]["b"]["fidelity"]) <= .05
                                      for row in results.values())
    p3 = all(all(row["budgets"]["32"][name]["fidelity"]
                 >= row["budgets"]["32"]["random_p95"] + .20 for name in ("a", "b"))
             for row in results.values())
    p4 = all(all(row["budgets"]["16"][name]["fidelity"] >= row["full"]["fidelity"] - .05
                 for name in ("a", "b")) for row in results.values())
    hypotheses = {
        "P0_top32_graph_is_selection_stable": {"supported": p0, "overlaps": overlaps},
        "P1_both_top32_graphs_are_sufficient": {"supported": p1},
        "P2_function_is_more_stable_than_edges": {"supported": p2},
        "P3_selected_graphs_beat_random_null": {"supported": p3},
        "P4_top16_is_sufficient": {"supported": p4},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "training": training,
                                  "hypotheses": hypotheses, "overlaps": overlaps,
                                  "results": results}, indent=2, allow_nan=False))
    lines = ["# Replication red team of the sparse pair graph", "",
             "The natural rank-32 basis and weight-only joint rotation are held fixed. Quadratic coordinate pairs are ranked independently on two disjoint 256-example FineWeb panels, frozen, and evaluated on FineWeb plus two OOD corpora. Twenty matched random graphs estimate a null distribution.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Pair-set Jaccard at 16/32/64: `{overlaps['16']:.3f}/{overlaps['32']:.3f}/{overlaps['64']:.3f}`.", "",
              "| Corpus | Full | A/B at 16 | A/B at 32 | A/B at 64 | Random p95 at 32 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        pair = lambda count: f"{row['budgets'][str(count)]['a']['fidelity']:.3f}/{row['budgets'][str(count)]['b']['fidelity']:.3f}"
        lines.append(f"| {corpus} | {row['full']['fidelity']:.3f} | {pair(16)} | {pair(32)} | {pair(64)} | "
                     f"{row['budgets']['32']['random_p95']:.3f} |")
    lines += ["", "Stable function with unstable exact edges would identify an equivalence class of short interaction programs rather than one privileged circuit graph. Stable edges and function would justify inspecting individual coordinate pairs semantically. Failure against the random null would retract the sparse-graph claim."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
