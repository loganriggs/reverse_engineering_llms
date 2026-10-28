#!/usr/bin/env python3
"""Swap complete valid final-MLP address codewords across functional classes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_mixed_weight_canonical_address_subspace import tensor_write


CONFIG = Path("configs/grammar_mixed_address_codeword_swaps.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_address_codeword_swaps.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ADDRESS_CODEWORD_SWAPS.md")


def donors(relations, labels, relation_delta, answer_delta, roll=0):
    groups = {(r, a): np.flatnonzero((relations == r) & (labels == a)) for r in range(8) for a in range(8)}
    counters = {(r, a): 0 for r in range(8) for a in range(8)}
    output = np.empty(len(labels), dtype=np.int64)
    for i, (relation, answer) in enumerate(zip(relations, labels)):
        key = ((int(relation) + relation_delta) % 8, (int(answer) + answer_delta) % 8)
        choices = groups[key]
        output[i] = choices[(counters[key] + roll) % len(choices)]
        counters[key] += 1
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
        write_table = prototypes(train["write"], facts[train_idx, 2], labels[train_idx], True)
        output[status] = {}
        for template in config["evaluation_templates"]:
            data, _, _ = collect(model, encode_prompts(manifest, facts[eval_idx], template), device)
            relations = facts[eval_idx, 2]
            answers = labels[eval_idx]
            oracle_write = apply_prototypes(write_table, relations, answers, True)
            index = {
                "native": np.arange(len(eval_idx)),
                "same_cell": donors(relations, answers, 0, 0, roll=1),
                "same_answer_next_relation": donors(relations, answers, 1, 0),
                "next_answer_same_relation": donors(relations, answers, 0, 1),
                "next_answer_next_relation": donors(relations, answers, 1, 1),
            }
            methods = {}
            for name, donor_idx in index.items():
                write = tensor_write(model, data["premlp"][torch.as_tensor(donor_idx)])
                row = score(write, data["write"], data["after"], data["zero_logits"], data["logits"], answers, oracle_write, model, config["minimum_fisher_effect_norm"])
                donor_answers = answers[donor_idx]
                routed = score(write, data["write"], data["after"], data["zero_logits"], data["logits"], donor_answers, oracle_write, model, config["minimum_fisher_effect_norm"])
                row["donor_answer_accuracy"] = routed["answer_accuracy"]
                methods[name] = row
            output[status][str(template)] = methods
    return {"seed": spec["seed"], "results": output}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [analyze(spec, config, device) for spec in config["models"]]
    first, held = map(str, config["evaluation_templates"])
    def val(run, status, template, method, key="median_effect_fidelity"):
        return run["results"][status][template][method][key]
    q0 = all(val(run,status,template,"native") >= .9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    q1 = all(val(run,status,held,"same_cell") >= .95 for run in runs for status in ("selected","unselected"))
    q2 = all(val(run,status,held,"same_answer_next_relation") >= .90 for run in runs for status in ("selected","unselected"))
    q3 = all(val(run,"selected",held,"next_answer_same_relation","donor_answer_accuracy") >= .80 for run in runs)
    q4 = all(val(run,"selected",held,"next_answer_next_relation","donor_answer_accuracy") >= .80 for run in runs)
    q5 = all(val(run,"selected",held,"next_answer_same_relation","answer_accuracy") <= .30 for run in runs)
    q6 = all(abs(val(run,status,held,"next_answer_same_relation","donor_answer_accuracy") - val(run,status,held,"next_answer_next_relation","donor_answer_accuracy")) <= .10 for run in runs for status in ("selected","unselected"))
    q7 = all(abs(val(run,"selected",held,"same_answer_next_relation") - val(run,"unselected",held,"same_answer_next_relation")) <= .10 for run in runs)
    q8 = all(abs(val(run,"selected",first,"same_answer_next_relation") - val(run,"selected",held,"same_answer_next_relation")) <= .10 for run in runs)
    q9 = all(row["effectful_examples"] >= 64 for run in runs for status in ("selected","unselected") for template in (first,held) for row in run["results"][status][template].values())
    hypotheses = {name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(q0,q1,q2,q3,q4,q5,q6,q7,q8,q9))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Valid mixed-address codeword swaps","","Complete normalized states from real evaluation examples are swapped before the original final bilinear tensor. `answer_accuracy` scores the recipient's original answer; `donor_answer_accuracy` scores the answer associated with the donor state.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | same-cell fid | same-answer/new-relation fid | next-answer/same-relation target/donor acc | next-answer/new-relation target/donor acc |","|---:|---|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            same=val(run,status,held,"next_answer_same_relation","answer_accuracy"); same_d=val(run,status,held,"next_answer_same_relation","donor_answer_accuracy")
            cross=val(run,status,held,"next_answer_next_relation","answer_accuracy"); cross_d=val(run,status,held,"next_answer_next_relation","donor_answer_accuracy")
            lines.append(f"| {run['seed']} | {status} | {val(run,status,held,'same_cell'):.3f} | {val(run,status,held,'same_answer_next_relation'):.3f} | {same:.3f}/{same_d:.3f} | {cross:.3f}/{cross_d:.3f} |")
    lines += ["","These are quotient tests: high same-answer fidelity means many geometrically different states implement the same downstream operation. It does not imply that the upstream model computes those states in the same way."]
    REPORT.write_text("\n".join(lines)+"\n")
    print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__": main()
