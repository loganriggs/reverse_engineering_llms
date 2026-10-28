#!/usr/bin/env python3
"""Execute PCA-projected address states through the native bilinear tensor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel


CONFIG = Path("configs/grammar_mixed_weight_canonical_address_subspace.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_weight_canonical_address_subspace.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_WEIGHT_CANONICAL_ADDRESS_SUBSPACE.md")


def pca(train):
    mean = train.mean(0); _, _, vh = torch.linalg.svd(train - mean, full_matrices=False)
    return mean, vh


def project(values, mean, basis):
    return mean + (values - mean) @ basis.T @ basis


def tensor_write(model, state):
    block = model.blocks[3]
    return ((state @ block.left.weight.detach().cpu().T) *
            (state @ block.right.weight.detach().cpu().T)) @ block.down.weight.detach().cpu().T


def analyze(spec, config, device):
    root=Path(spec["root"]); manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]); labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=np.ones(len(facts),bool)
    for template in config["evaluation_templates"]:
        prompts=encode_prompts(manifest,facts,template)
        for start in range(0,len(prompts),1024):
            tokens=torch.as_tensor(prompts[start:start+1024].astype(np.int64),device=device)
            prediction=model(tokens)[:,-1,list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start+len(prediction)] &= prediction==labels[start:start+len(prediction)]
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"])
    output={}
    for status_bool,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[status_bool]["discovery"]; eval_idx=panel[status_bool]["evaluation"]
        train,_,_=collect(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),device)
        mean,basis=pca(train["premlp"])
        relation_proto=prototypes(train["write"],facts[train_idx,2],labels[train_idx],True)
        output[status]={}
        for template in config["evaluation_templates"]:
            evaluation,_,_=collect(model,encode_prompts(manifest,facts[eval_idx],template),device)
            proto=apply_prototypes(relation_proto,facts[eval_idx,2],labels[eval_idx],True)
            methods={}
            for rank in config["ranks"]:
                state=project(evaluation["premlp"],mean,basis[:rank]); prediction=tensor_write(model,state)
                methods[f"pca_rank_{rank}"]=score(prediction,evaluation["write"],evaluation["after"],evaluation["zero_logits"],evaluation["logits"],labels[eval_idx],proto,model,config["minimum_fisher_effect_norm"])
            generator=torch.Generator().manual_seed(config["seed"]+spec["seed"]*100+int(status_bool)*10+template)
            random_fidelities=[]
            for repeat in range(config["random_subspaces"]):
                random_basis=torch.linalg.qr(torch.randn(model.width,48,generator=generator)).Q.T
                prediction=tensor_write(model,project(evaluation["premlp"],mean,random_basis))
                row=score(prediction,evaluation["write"],evaluation["after"],evaluation["zero_logits"],evaluation["logits"],labels[eval_idx],proto,model,config["minimum_fisher_effect_norm"])
                random_fidelities.append(row["median_effect_fidelity"])
            methods["random_rank48"]={"median_fidelity_mean":float(np.mean(random_fidelities)),"median_fidelity_p95":float(np.quantile(random_fidelities,.95)),"replicates":len(random_fidelities)}
            output[status][str(template)]=methods
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu";runs=[]
    for spec in config["models"]: print("project",spec["seed"],flush=True);runs.append(analyze(spec,config,device))
    held=str(config["evaluation_templates"][-1]); first=str(config["evaluation_templates"][0])
    def get(run,status,template,rank,key="median_effect_fidelity"): return run["results"][status][template][f"pca_rank_{rank}"][key]
    p0=all(get(run,status,template,96)>=.9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    p1=all(get(run,"selected",held,48)>=.8 for run in runs)
    p2=all(get(run,"selected",held,32)>=.7 for run in runs)
    p3=all(get(run,"selected",held,64)<get(run,"selected",held,48)+.05 for run in runs)
    p4=all(get(run,"selected",held,48)>=run["results"]["selected"][held]["random_rank48"]["median_fidelity_p95"]+.15 for run in runs)
    p5=all(get(run,"selected",held,48,"oracle_relation_answer_residual_r2")>=.5 for run in runs)
    p6=all(get(run,"selected",held,48)>=get(run,"unselected",held,48)-.15 for run in runs)
    p7=all(abs(get(run,"selected",first,48)-get(run,"selected",held,48))<=.1 for run in runs)
    p8=all(run["results"][status][template][f"pca_rank_{rank}"]["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for template in (first,held) for rank in config["ranks"])
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(p0,p1,p2,p3,p4,p5,p6,p7,p8))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Weight-canonical causal subspace for the mixed address code","",
           "PCA is fit only to discovery-template normalized pre-MLP states. Held-out states are projected into each frozen subspace and executed through the original left/right/down bilinear weights—no write regression is fit. Equal-rank random subspaces use the same state mean and native tensor.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | Template | "+" | ".join(f"r{r}" for r in config["ranks"])+" | random48 mean/p95 |","|---:|---|---:|"+"---:|"*len(config["ranks"])+"---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            for template in (first,held):
                random=run["results"][status][template]["random_rank48"]
                lines.append(f"| {run['seed']} | {status} | {template} | "+" | ".join(f"{get(run,status,template,r):.3f}" for r in config["ranks"])+f" | {random['median_fidelity_mean']:.3f}/{random['median_fidelity_p95']:.3f} |")
    lines += ["","This is a causal compression test of the fixed tensor program. Success identifies a reusable state subspace; it does not make the PCA axes individually semantic or explain how upstream layers place each fact in that subspace."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
