#!/usr/bin/env python3
"""Permutation distribution for the causal final-gate answer quotient."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_final_gate_code_dimension import gate, project, eligible_mask
from run_grammar_final_gate_answer_quotient import answer_table, centroid_basis


CONFIG=Path("configs/grammar_final_gate_answer_null_distribution.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_final_gate_answer_null_distribution.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_FINAL_GATE_ANSWER_NULL_DISTRIBUTION.md")


def analyze(spec,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,[config["discovery_template"],config["evaluation_template"]],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"])
    down=model.blocks[3].down.weight.detach().cpu();output={}
    for flag,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[flag]["discovery"];eval_idx=panel[flag]["evaluation"]
        train,_,_=collect(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),device)
        data,_,_=collect(model,encode_prompts(manifest,facts[eval_idx],config["evaluation_template"]),device)
        train_gate=gate(model,train["premlp"]);native=gate(model,data["premlp"])
        write_table=prototypes(train["write"],facts[train_idx,2],labels[train_idx],True)
        oracle=apply_prototypes(write_table,facts[eval_idx,2],labels[eval_idx],True)
        def assess(label_assignment):
            table=answer_table(train_gate,label_assignment);mean,basis,_=centroid_basis(table)
            candidate=project(native,mean,basis)
            return score(candidate@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])
        real=assess(labels[train_idx]);null=[];rng=np.random.default_rng(config["seed"]+spec["seed"]*10+int(flag))
        for _ in range(config["permutations"]):
            shuffled=np.asarray(labels[train_idx]).copy();rng.shuffle(shuffled);null.append(assess(shuffled))
        fidelities=np.asarray([row["median_effect_fidelity"] for row in null])
        output[status]={"real":real,"null_mean":float(fidelities.mean()),"null_p95":float(np.quantile(fidelities,.95)),
                        "null_p99":float(np.quantile(fidelities,.99)),"null_max":float(fidelities.max()),
                        "empirical_p":float((1+np.sum(fidelities>=real["median_effect_fidelity"]))/(1+len(fidelities))),
                        "minimum_null_effectful_examples":int(min(row["effectful_examples"] for row in null)),
                        "null_fidelities":fidelities.tolist()}
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu";runs=[analyze(spec,config,device) for spec in config["models"]]
    n0=all(row["real"]["median_effect_fidelity"]-row["null_p99"]>=.40 for run in runs for row in run["results"].values())
    n1=all(row["empirical_p"]<=.01 for run in runs for row in run["results"].values())
    n2=all(row["null_mean"]<.35 for run in runs for row in run["results"].values())
    n3=all(row["real"]["median_effect_fidelity"]>=.90 for run in runs for row in run["results"].values())
    n4=all(row["real"]["effectful_examples"]>=64 and row["minimum_null_effectful_examples"]>=64 for run in runs for row in run["results"].values())
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(n0,n1,n2,n3,n4))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Permutation null for the final-gate answer quotient","","One hundred balanced-label permutations are fit on the same discovery gates and evaluated by finite execution through native weights. The empirical p-value uses the standard plus-one correction.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | real | null mean/p95/p99/max | empirical p |","|---:|---|---:|---:|---:|"]
    for run in runs:
        for status,row in run["results"].items():lines.append(f"| {run['seed']} | {status} | {row['real']['median_effect_fidelity']:.3f} | {row['null_mean']:.3f}/{row['null_p95']:.3f}/{row['null_p99']:.3f}/{row['null_max']:.3f} | {row['empirical_p']:.4f} |")
    lines += ["","This validates the semantic orientation of the seven-dimensional quotient against the tested permutation family. It does not make its internal basis unique or explain upstream retrieval."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
