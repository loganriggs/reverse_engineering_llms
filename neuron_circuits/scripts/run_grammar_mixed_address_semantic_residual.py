#!/usr/bin/env python3
"""Finite decomposition of mixed address states into semantic prototypes and residuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_mixed_weight_canonical_address_subspace import tensor_write


CONFIG = Path("configs/grammar_mixed_address_semantic_residual.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_address_semantic_residual.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ADDRESS_SEMANTIC_RESIDUAL.md")


def grouped_roll(values, relations, labels):
    output = values.clone()
    touched = torch.zeros(len(values), dtype=torch.bool)
    for relation in range(8):
        for label in range(8):
            idx = torch.as_tensor(np.flatnonzero((relations == relation) & (labels == label)))
            if len(idx) > 1:
                output[idx] = values[idx.roll(1)]
                touched[idx] = True
    if not bool(touched.all()):
        raise RuntimeError(f"grouped residual roll left {int((~touched).sum())} examples untouched")
    return output


def analyze(spec, config, device):
    root = Path(spec["root"])
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"])
    labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = np.ones(len(facts), bool)
    for template in config["evaluation_templates"]:
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            pred = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(pred)] &= pred == labels[start:start + len(pred)]
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    output = {}
    for flag, status in ((False, "unselected"), (True, "selected")):
        train_idx = panel[flag]["discovery"]
        eval_idx = panel[flag]["evaluation"]
        train, _, _ = collect(model, encode_prompts(manifest, facts[train_idx], config["discovery_template"]), device)
        state_table = prototypes(train["premlp"], facts[train_idx, 2], labels[train_idx], True)
        write_table = prototypes(train["write"], facts[train_idx, 2], labels[train_idx], True)
        global_mean = train["premlp"].mean(0)
        output[status] = {}
        for template in config["evaluation_templates"]:
            data, _, _ = collect(model, encode_prompts(manifest, facts[eval_idx], template), device)
            relations = facts[eval_idx, 2]
            answers = labels[eval_idx]
            proto = apply_prototypes(state_table, relations, answers, True)
            oracle_write = apply_prototypes(write_table, relations, answers, True)
            residual = data["premlp"] - proto
            within = grouped_roll(residual, relations, answers)
            generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 100 + int(flag) * 10 + template)
            global_shuffle = residual[torch.randperm(len(residual), generator=generator)]
            wrong_answers = (answers + 1) % 8
            wrong_proto = apply_prototypes(state_table, relations, wrong_answers, True)
            states = {
                "native": data["premlp"],
                "prototype_only": proto,
                "residual_only": global_mean + residual,
                "within_cell_residual_swap": proto + within,
                "global_residual_shuffle": proto + global_shuffle,
                "half_residual": proto + .5 * residual,
                "one_and_half_residual": proto + 1.5 * residual,
                "wrong_answer_prototype": wrong_proto + residual,
            }
            methods = {}
            for name, state in states.items():
                write = tensor_write(model, state)
                methods[name] = score(write, data["write"], data["after"], data["zero_logits"], data["logits"], answers, oracle_write, model, config["minimum_fisher_effect_norm"])
            output[status][str(template)] = methods
    return {"seed": spec["seed"], "results": output}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [analyze(spec, config, device) for spec in config["models"]]
    first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, method, key="median_effect_fidelity"):
        return run["results"][status][template][method][key]
    d0 = all(val(run, status, template, "native") >= .9999 for run in runs for status in ("selected", "unselected") for template in (first, held))
    d1 = all(val(run, "selected", held, "prototype_only", "answer_accuracy") >= .95 and val(run, "selected", held, "prototype_only") < .80 for run in runs)
    d2 = all(val(run, "selected", held, "residual_only", "answer_accuracy") <= .60 for run in runs)
    d3 = all(val(run, "selected", held, "within_cell_residual_swap", "answer_accuracy") >= .90 for run in runs)
    d4 = all(val(run, "selected", held, "native") - val(run, "selected", held, "within_cell_residual_swap") >= .15 for run in runs)
    d5 = all(val(run, "selected", held, "within_cell_residual_swap") - val(run, "selected", held, "global_residual_shuffle") >= .10 for run in runs)
    d6 = all(val(run, "selected", held, "half_residual", "answer_accuracy") >= .90 for run in runs)
    d7 = all(val(run, "selected", held, "wrong_answer_prototype", "answer_accuracy") < .30 for run in runs)
    d8 = all(abs(val(run, "selected", held, "within_cell_residual_swap") - val(run, "unselected", held, "within_cell_residual_swap")) <= .15 for run in runs)
    d9 = all(abs(val(run, "selected", first, "within_cell_residual_swap") - val(run, "selected", held, "within_cell_residual_swap")) <= .10 for run in runs)
    d10 = all(row["effectful_examples"] >= 64 for run in runs for status in ("selected", "unselected") for template in (first, held) for row in run["results"][status][template].values())
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10))}
    correction = {"date": "2026-10-20", "issue": "The first execution iterated over only relations 0-3 in the within-cell residual swap, leaving half the examples native and forcing a misleading median fidelity of approximately 1. This rerun covers all eight relations and asserts full coverage. Locked hypotheses were not changed."}
    OUTPUT.write_text(json.dumps({"config": config, "correction": correction, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Semantic-prototype versus residual address interventions", "", "**Correction:** the first execution covered only relations 0–3 in the within-cell swap, leaving half the panel unchanged and producing a spurious median near 1. This rerun covers all eight relations, asserts full coverage, and retains the original locked hypotheses.", "", "Each state is executed through the original bilinear tensor. `prototype_only` uses an oracle relation-answer mean learned on a disjoint discovery panel. Residual swaps therefore test precise state construction, not whether the answer can be guessed without the prompt.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Status | proto fid/acc | residual fid/acc | within-swap fid/acc | global-swap fid/acc | half fid/acc | wrong-proto fid/acc |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected", "selected"):
            def pair(method):
                return f"{val(run,status,held,method):.3f}/{val(run,status,held,method,'answer_accuracy'):.3f}"
            lines.append(f"| {run['seed']} | {status} | {pair('prototype_only')} | {pair('residual_only')} | {pair('within_cell_residual_swap')} | {pair('global_residual_shuffle')} | {pair('half_residual')} | {pair('wrong_answer_prototype')} |")
    lines += ["", "Accuracy asks whether the final answer survives; effect fidelity asks whether the full eight-answer causal correction is reproduced. Their divergence is itself mechanistic evidence: an answer actuator can be behaviorally sufficient while omitting much of the native computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
