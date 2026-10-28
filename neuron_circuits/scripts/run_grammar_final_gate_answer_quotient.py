#!/usr/bin/env python3
"""Test whether the compact final gate code is principally an answer quotient."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import collect, prototypes, apply_prototypes, score, select_panel
from run_grammar_final_gate_code_dimension import gate, pca, project, eligible_mask


CONFIG=Path("configs/grammar_final_gate_answer_quotient.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_final_gate_answer_quotient.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_FINAL_GATE_ANSWER_QUOTIENT.md")


def answer_table(values, labels):
    return torch.stack([values[torch.as_tensor(labels==answer)].mean(0) for answer in range(8)])


def centroid_basis(table):
    centered=table-table.mean(0);_,singular,basis=torch.linalg.svd(centered,full_matrices=False)
    rank=int((singular>singular.max()*1e-6).sum())
    return table.mean(0),basis[:rank],rank


def analyze(spec,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"])
    down=model.blocks[3].down.weight.detach().cpu();output={}
    for flag,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[flag]["discovery"];eval_idx=panel[flag]["evaluation"]
        train,_,_=collect(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),device)
        train_gate=gate(model,train["premlp"]);global_mean=train_gate.mean(0)
        answer_proto=answer_table(train_gate,labels[train_idx]);answer_mean,answer_basis,answer_rank=centroid_basis(answer_proto)
        relation_answer_gate=prototypes(train_gate,facts[train_idx,2],labels[train_idx],True)
        write_table=prototypes(train["write"],facts[train_idx,2],labels[train_idx],True)
        _,pca_basis=pca(train_gate)
        generator=torch.Generator().manual_seed(config["seed"]+spec["seed"]+int(flag))
        shuffled_labels=np.asarray(labels[train_idx]).copy();np.random.default_rng(config["seed"]+spec["seed"]+int(flag)).shuffle(shuffled_labels)
        shuffled_proto=answer_table(train_gate,shuffled_labels);shuffled_mean,shuffled_basis,shuffled_rank=centroid_basis(shuffled_proto)
        centered=train_gate-global_mean
        between=answer_proto[torch.as_tensor(labels[train_idx],dtype=torch.long)]-global_mean
        variance_fraction=float(between.square().sum()/centered.square().sum().clamp_min(1e-30))
        output[status]={"metadata":{"answer_span_rank":answer_rank,"shuffled_span_rank":shuffled_rank,"between_answer_variance_fraction":variance_fraction}}
        for template in config["evaluation_templates"]:
            data,_,_=collect(model,encode_prompts(manifest,facts[eval_idx],template),device)
            native=gate(model,data["premlp"]);answers=labels[eval_idx];relations=facts[eval_idx,2]
            oracle=apply_prototypes(write_table,relations,answers,True)
            answer_state=answer_proto[torch.as_tensor(answers,dtype=torch.long)]
            relation_answer_state=apply_prototypes(relation_answer_gate,relations,answers,True)
            states={
                "native":native,
                "pca_8":project(native,global_mean,pca_basis[:8]),
                "answer_span":project(native,answer_mean,answer_basis),
                "answer_prototype":answer_state,
                "relation_answer_prototype":relation_answer_state,
                "within_answer_residual":global_mean+(native-answer_state),
                "shuffled_answer_span":project(native,shuffled_mean,shuffled_basis),
            }
            methods={}
            for name,state in states.items():
                methods[name]=score(state@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],answers,oracle,model,config["minimum_fisher_effect_norm"])
            output[status][str(template)]=methods
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu"
    runs=[analyze(spec,config,device) for spec in config["models"]];first,held=map(str,config["evaluation_templates"])
    def val(run,status,template,method,key="median_effect_fidelity"):return run["results"][status][template][method][key]
    e0=all(val(run,status,template,"native")>=.9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    e1=all(val(run,"selected",held,"answer_span")>=.70 for run in runs)
    e2=all(val(run,"selected",held,"pca_8")-val(run,"selected",held,"answer_span")<=.20 for run in runs)
    e3=all(val(run,"selected",held,"answer_prototype","answer_accuracy")>=.90 for run in runs)
    e4=all(val(run,"selected",held,"answer_prototype")<.80 for run in runs)
    e5=all(val(run,"selected",held,"relation_answer_prototype","answer_accuracy")-val(run,"selected",held,"answer_prototype","answer_accuracy")<=.05 for run in runs)
    e6=all(val(run,"selected",held,"relation_answer_prototype")-val(run,"selected",held,"answer_prototype")>=.05 for run in runs)
    e7=all(val(run,status,held,"shuffled_answer_span")<.30 for run in runs for status in ("selected","unselected"))
    e8=all(val(run,"selected",held,"within_answer_residual","answer_accuracy")<=.60 for run in runs)
    e9=all(abs(val(run,"selected",held,"answer_span")-val(run,"unselected",held,"answer_span"))<=.15 for run in runs)
    e10=all(abs(val(run,"selected",first,"answer_span")-val(run,"selected",held,"answer_span"))<=.10 for run in runs)
    e11=all(row["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for template in (first,held) for row in run["results"][status][template].values())
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Is the compact final gate code an answer quotient?","","The answer-centroid span is learned on a disjoint discovery panel and has rank at most seven. All gate candidates use the native down projection and exact suffix. Oracle prototypes are actuator diagnostics, not explanations of how the prompt retrieves its answer.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | answer variance | PCA8 fid | answer-span fid/acc | answer proto fid/acc | relation+answer proto fid/acc | residual fid/acc | shuffled span |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            def pair(method):return f"{val(run,status,held,method):.3f}/{val(run,status,held,method,'answer_accuracy'):.3f}"
            meta=run["results"][status]["metadata"]
            lines.append(f"| {run['seed']} | {status} | {meta['between_answer_variance_fraction']:.3f} | {val(run,status,held,'pca_8'):.3f} | {pair('answer_span')} | {pair('answer_prototype')} | {pair('relation_answer_prototype')} | {pair('within_answer_residual')} | {val(run,status,held,'shuffled_answer_span'):.3f} |")
    lines += ["","A successful answer quotient explains what the dense gate code controls. It does not explain the upstream lookup that chooses a point in that quotient, and prototype success cannot be used as a non-oracle retrieval result."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
