#!/usr/bin/env python3
"""Test whether compact late write modules generalize beyond newline edits."""

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


CONFIG = Path("configs/bilin12_multi_operation_modules.json")
OUTPUT = TENSOR_ROOT / "bilin12_multi_operation_modules.json"
REPORT = TENSOR_ROOT / "BILIN12_MULTI_OPERATION_MODULES.md"


def activation_delta(block, panel):
    pre, _, _, changed_pre, _ = panel
    dimension = pre.shape[-1]
    base = F.rms_norm(pre, (dimension,)); changed = F.rms_norm(changed_pre, (dimension,))
    return (block.mlp.Left(changed) * block.mlp.Right(changed)
            - block.mlp.Left(base) * block.mlp.Right(base))


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    cooperation = json.loads(Path(config["parent_config"]).read_text())
    analytic = json.loads(Path(cooperation["parent_config"]).read_text())
    subspace = json.loads(Path(analytic["parent_config"]).read_text())
    state_config = json.loads(Path(subspace["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    tokens = {name: token_id(tokenizer, name) for name in config["targets"]}
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    block = model.blocks[parent["layer"]]; down = block.mlp.Down.weight.float()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_basis = torch.linalg.qr(torch.randn(model.config.n_embd, config["rank"],
                                               generator=generator, device="cuda")).Q
    random_channels = torch.randperm(down.shape[1], generator=generator,
                                     device="cuda")[:config["channels"]]
    modules = {}
    for name, target in tokens.items():
        panel = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                             target, parent["batch_size"])
        delta = panel[4] - panel[1]
        _, _, vh = torch.linalg.svd(delta.float(), full_matrices=False)
        basis = vh[:config["rank"]].T.contiguous()
        act = activation_delta(block, panel)
        contribution_norm = (down.T @ basis).norm(dim=-1)
        importance = (act.abs() * contribution_norm).mean(0)
        modules[name] = {"basis": basis, "channels": importance.argsort(descending=True)[:config["channels"]]}

    names = list(tokens)
    overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f"{left!r}__{right!r}"] = {
                "subspace": float((modules[left]["basis"].T @ modules[right]["basis"]).square().sum()
                                  / config["rank"]),
                "channel_jaccard": float(len(set(modules[left]["channels"].tolist())
                                             & set(modules[right]["channels"].tolist()))
                                         / len(set(modules[left]["channels"].tolist())
                                               | set(modules[right]["channels"].tolist()))),
            }
    results = {}
    for index, name in enumerate(names):
        target = tokens[name]; wrong_name = names[(index + 1) % len(names)]
        basis = modules[name]["basis"]; wrong_basis = modules[wrong_name]["basis"]
        correct_channels = modules[name]["channels"]
        wrong_channels = modules[wrong_name]["channels"]
        rows = {}
        for corpus, spec in parent["evaluation"].items():
            print("evaluate", repr(name), corpus, flush=True)
            panel = paired_panel(model, spec, parent["layer"], target, parent["batch_size"])
            delta = panel[4] - panel[1]; act = activation_delta(block, panel)
            project = lambda value, use_basis: (value @ use_basis) @ use_basis.T
            correct_write = project(delta, basis)
            wrong_write = project(delta, wrong_basis)
            random_write = project(delta, random_basis)
            def channel_write(indices):
                selected = torch.zeros_like(act); selected[:, indices] = act[:, indices]
                return project(block.mlp.Down(selected), basis)
            rows[corpus] = {
                "examples": len(delta), "correct_subspace": causal(model, panel, correct_write),
                "wrong_subspace": causal(model, panel, wrong_write),
                "random_subspace": causal(model, panel, random_write),
                "correct_channels": causal(model, panel, channel_write(correct_channels)),
                "wrong_channels": causal(model, panel, channel_write(wrong_channels)),
                "random_channels": causal(model, panel, channel_write(random_channels)),
            }
        results[name] = {"token_id": target, "wrong_operation": wrong_name, "corpora": rows}
    m0 = all(row["correct_subspace"]["fidelity"] >= .70
             for operation in results.values() for row in operation["corpora"].values())
    m1 = all(row["correct_subspace"]["fidelity"]
             >= row["random_subspace"]["fidelity"] + .30
             for operation in results.values() for row in operation["corpora"].values())
    m2 = all(np.mean([row["correct_subspace"]["fidelity"]
                      - row["wrong_subspace"]["fidelity"]
                      for row in operation["corpora"].values()]) >= .15
             for operation in results.values())
    m3 = all(np.mean([row["correct_channels"]["fidelity"]
                      - row["wrong_channels"]["fidelity"]
                      for row in operation["corpora"].values()]) >= .10
             for operation in results.values())
    m4 = all(row["correct_channels"]["fidelity"] >= .60
             for operation in results.values() for row in operation["corpora"].values())
    hypotheses = {
        "M0_compact_write_modules_generalize_across_operations": {"supported": m0},
        "M1_learned_modules_beat_random": {"supported": m1},
        "M2_write_modules_are_operation_specific": {"supported": m2},
        "M3_native_channel_banks_are_operation_specific": {"supported": m3},
        "M4_fixed_native_banks_are_broadly_useful": {"supported": m4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "tokens": tokens,
                                  "hypotheses": hypotheses, "overlap": overlap,
                                  "results": results}, indent=2))
    lines = ["# Multi-operation late write modules", "",
             "Rank-32 write subspaces and top-128 native channel banks are learned separately for four token replacement operations on FineWeb and frozen before three-corpus evaluation.", "", "## Locked hypotheses", ""]
    for key, item in hypotheses.items():
        lines.append(f"- **{key}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Target | Corpus | Correct/wrong/random subspace fidelity | Correct/wrong/random channel fidelity |",
              "|---|---|---:|---:|"]
    for name, operation in results.items():
        for corpus, row in operation["corpora"].items():
            lines.append(
                f"| `{name.encode('unicode_escape').decode()}` | {corpus} | "
                f"{row['correct_subspace']['fidelity']:.3f}/{row['wrong_subspace']['fidelity']:.3f}/{row['random_subspace']['fidelity']:.3f} | "
                f"{row['correct_channels']['fidelity']:.3f}/{row['wrong_channels']['fidelity']:.3f}/{row['random_channels']['fidelity']:.3f} |"
            )
    mean_subspace = np.mean([row["subspace"] for row in overlap.values()])
    mean_jaccard = np.mean([row["channel_jaccard"] for row in overlap.values()])
    lines += ["", f"Mean pairwise subspace overlap: `{mean_subspace:.3f}`; mean top-128 channel Jaccard: `{mean_jaccard:.3f}`.", "",
              "Correct-vs-wrong comparisons test operation specificity. High cross-operation transfer would instead support one generic late edit interface shared by several token changes."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
