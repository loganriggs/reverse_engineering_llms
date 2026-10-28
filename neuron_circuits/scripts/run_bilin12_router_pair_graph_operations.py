#!/usr/bin/env python3
"""Test whether sparse router edges are shared across token-edit operations."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import effect_similarity, logits_from_hidden, token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_router_joint_diagonalization import evaluate_forms, masked_forms
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_pair_graph_operations.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_pair_graph_operations.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_PAIR_GRAPH_OPERATIONS.md"


def pair_scores(forms, base, edit):
    rows = []
    for i in range(forms.shape[-1]):
        for j in range(i + 1, forms.shape[-1]):
            scalar = 2 * (base[:, i] * edit[:, j] + base[:, j] * edit[:, i]
                          + edit[:, i] * edit[:, j])
            contribution = scalar[:, None] * forms[:, i, j][None]
            rows.append(((i, j), float(contribution.square().sum(-1).mean())))
    return dict(rows)


def coordinates(model, panel, basis, rotation):
    dimension = model.config.n_embd
    base = F.rms_norm(panel[0], (dimension,)).float() @ basis @ rotation
    changed = F.rms_norm(panel[3], (dimension,)).float() @ basis @ rotation
    return base, changed - base


def relative_module(model, panel, basis, full_coefficients, predicted_coefficients):
    pre, write, base_logits, _, _ = panel
    full_logits = logits_from_hidden(model, pre + write + full_coefficients @ basis.T)
    predicted_logits = logits_from_hidden(model, pre + write + predicted_coefficients @ basis.T)
    cosine, fidelity = effect_similarity(base_logits, full_logits - base_logits,
                                          predicted_logits - base_logits)
    return {"cosine": cosine, "fidelity": fidelity}


def jaccard(left, right, count):
    a, b = set(left[:count]), set(right[:count])
    return len(a & b) / len(a | b)


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    targets = {name: token_id(tokenizer, name) for name in config["targets"]}
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural_states, _ = collect_states_writes(model, 11, config["natural"], config["positions"],
                                               config["batch_size"])
    _, _, vh = torch.linalg.svd(natural_states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms,
                              natural_basis).detach()
    rotation, training = train_rotation(restricted, config); rotation = rotation.detach()
    forms = rotation.T[None] @ restricted @ rotation[None]
    with torch.inference_mode():
        scores = {}; orders = {}
        for name, target in targets.items():
            panel = paired_panel(model, config["graph_discovery"], 11, target, config["batch_size"])
            base, edit = coordinates(model, panel, natural_basis, rotation)
            scores[name] = pair_scores(forms, base, edit)
            orders[name] = sorted(scores[name], key=scores[name].get, reverse=True)
        all_pairs = list(scores[next(iter(scores))])
        pooled_scores = {pair: float(np.mean([scores[name][pair] for name in scores]))
                         for pair in all_pairs}
        orders["pooled"] = sorted(all_pairs, key=pooled_scores.get, reverse=True)
        rng = np.random.default_rng(config["seed"]); orders["random"] = list(all_pairs)
        rng.shuffle(orders["random"])
        overlaps = {f"{left!r}__{right!r}": jaccard(orders[left], orders[right],
                                                     config["pair_budget"])
                    for left, right in combinations(targets, 2)}
        results = {}
        for name, target in targets.items():
            corpora = {}
            wrong = config["targets"][(config["targets"].index(name) + 1) % len(config["targets"])]
            for corpus, spec in config["evaluation"].items():
                print(repr(name), corpus, flush=True)
                panel = paired_panel(model, spec, 11, target, config["batch_size"])
                base, edit = coordinates(model, panel, natural_basis, rotation)
                full_coefficients = evaluate_forms(forms, base, edit)
                exact_coefficients = (panel[4] - panel[1]).float() @ write_basis
                methods = {}
                for label, order_name in (("own", name), ("pooled", "pooled"),
                                          ("newline", "\n"), ("wrong", wrong),
                                          ("random", "random")):
                    predicted = evaluate_forms(masked_forms(forms, orders[order_name],
                                                             config["pair_budget"]), base, edit)
                    methods[label] = {
                        "relative": relative_module(model, panel, write_basis,
                                                    full_coefficients, predicted),
                        "absolute": causal(model, panel, predicted @ write_basis.T),
                    }
                corpora[corpus] = {
                    "exact_oracle": causal(model, panel, exact_coefficients @ write_basis.T),
                    "restricted_full": causal(model, panel, full_coefficients @ write_basis.T),
                    "methods": methods,
                }
            results[name] = {"wrong_graph": wrong, "corpora": corpora}
    rows = [(name, corpus, row) for name, target_rows in results.items()
            for corpus, row in target_rows["corpora"].items()]
    o0 = all(row["methods"]["own"]["relative"]["fidelity"] >= .90 for _,_,row in rows)
    o1 = all(row["methods"]["pooled"]["relative"]["fidelity"] >= .85 for _,_,row in rows)
    o2 = all(np.mean([results[name]["corpora"][c]["methods"]["own"]["relative"]["fidelity"]
                      - results[name]["corpora"][c]["methods"]["newline"]["relative"]["fidelity"]
                      for c in config["evaluation"]]) >= .10
             for name in config["targets"] if name != "\n")
    o3 = all(value >= .50 for value in overlaps.values())
    o4 = all(all(row["methods"][method]["relative"]["fidelity"]
                     >= row["methods"]["random"]["relative"]["fidelity"] + .20
                     for method in ("own", "pooled")) for _,_,row in rows)
    o5 = all(row["methods"]["own"]["absolute"]["fidelity"]
             >= row["restricted_full"]["fidelity"] - .05 for _,_,row in rows)
    hypotheses = {
        "O0_own_graphs_recover_restricted_modules": {"supported": o0},
        "O1_one_pooled_graph_is_reusable": {"supported": o1},
        "O2_graphs_are_operation_specific": {"supported": o2},
        "O3_operation_graphs_share_a_backbone": {"supported": o3, "overlaps": overlaps},
        "O4_selected_graphs_beat_random": {"supported": o4},
        "O5_absolute_limit_is_output_not_graph": {"supported": o5},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "training": training,
                                  "hypotheses": hypotheses, "overlaps": overlaps,
                                  "results": results}, indent=2, allow_nan=False))
    lines = ["# Cross-operation test of the sparse router graph", "",
             "The pooled output basis, natural input space, and weight-only joint rotation are shared. Newline, period, comma, and ` the` each rank 32 coordinate pairs on the same FineWeb discovery rows; a pooled ranking averages contribution energy across operations. Relative fidelity treats the complete restricted rank-32 quadratic program as the target, separating graph compression from output-basis limitations.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Operation-graph Jaccards: `{json.dumps(overlaps)}`.", "",
              "| Target | Corpus | Exact/restricted absolute | Own/pooled/newline/wrong/random relative fidelity | Own/pooled absolute |",
              "|---|---|---:|---:|---:|"]
    for name, target_rows in results.items():
        for corpus, row in target_rows["corpora"].items():
            methods = row["methods"]
            relative = "/".join(f"{methods[key]['relative']['fidelity']:.3f}" for key in
                                 ("own", "pooled", "newline", "wrong", "random"))
            lines.append(f"| `{name.encode('unicode_escape').decode()}` | {corpus} | "
                         f"{row['exact_oracle']['fidelity']:.3f}/{row['restricted_full']['fidelity']:.3f} | "
                         f"{relative} | {methods['own']['absolute']['fidelity']:.3f}/"
                         f"{methods['pooled']['absolute']['fidelity']:.3f} |")
    lines += ["", "A shared graph would be a reusable interaction grammar atop behavior-specific coefficients. Operation-specific graphs would instead mean that the same compact coordinates are recombined by different sparse relational programs. Relative fidelity is intentionally conditional on the rank-32 module; absolute fidelity remains the test of whether that entire module is sufficient for the native edit."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
