#!/usr/bin/env python3
"""Matched finite gate-depth profiles for pure-rule and mixed-memory models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import select_panel
from run_grammar_final_gate_code_dimension import eligible_mask, pca, project, pure_panel
from run_grammar_final_gate_answer_quotient import answer_table, centroid_basis
from run_grammar_gate_bottleneck_depth_telescope import native_data, patched_data, metric


CONFIG=Path("configs/grammar_pure_mixed_gate_depth_comparison.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_pure_mixed_gate_depth_comparison.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_PURE_MIXED_GATE_DEPTH_COMPARISON.md")


def analyze(spec,regime,rho,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz");model=load_model(root/f"rho_{rho:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",rho)
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel_config=dict(config)
    for field in ("discovery_per_relation_answer_status", "validation_per_relation_answer_status",
                  "evaluation_per_relation_answer_status"):
        panel_config[field]=config.get(f"{regime}_{field}",config[field])
    if regime=="pure":panels={"all":pure_panel(labels,eligible,facts,panel_config,config["seed"]+spec["seed"])}
    else:
        raw=select_panel(labels,selected,changed,eligible,facts,panel_config,config["seed"]+spec["seed"]);panels={"unselected":raw[False],"selected":raw[True]}
    output={}
    for status,panel in panels.items():
        train_idx=panel["discovery"];eval_idx=panel["evaluation"]
        train=native_data(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),config["layers"],device)
        answer3_mean,answer3_basis,_=centroid_basis(answer_table(train["gates"][:,3],labels[train_idx]))
        fitted={}
        for layer in config["layers"]:
            values=train["gates"][:,layer];mean,basis=pca(values);answer_mean,answer_basis,_=centroid_basis(answer_table(values,labels[train_idx]))
            fitted[layer]=(mean,basis,answer_mean,answer_basis)
        output[status]={str(layer):{} for layer in config["layers"]}
        for template in config["evaluation_templates"]:
            prompts=encode_prompts(manifest,facts[eval_idx],template);data=native_data(model,prompts,config["layers"],device)
            for layer in config["layers"]:
                mean,basis,answer_mean,answer_basis=fitted[layer];native=data["gates"][:,layer];methods={}
                candidates={"native":native,"answer_span":project(native,answer_mean,answer_basis)}
                for rank in config["ranks"]:candidates[f"pca_{rank}"]=project(native,mean,basis[:rank])
                for name,candidate in candidates.items():
                    logits,gate3=patched_data(model,prompts,layer,candidate,device)
                    methods[name]=metric(logits,gate3,data,layer,labels[eval_idx],answer3_mean,answer3_basis,config["minimum_fisher_effect_norm"])
                methods["rank_k90"]=next((rank for rank in config["ranks"] if methods[f"pca_{rank}"]["median_effect_fidelity"]>=.90),193)
                output[status][str(layer)][str(template)]=methods
    return {"regime":regime,"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu"
    pure=[analyze(spec,"pure",config["pure_rho"],config,device) for spec in config["pure_models"]]
    mixed=[analyze(spec,"mixed",config["mixed_rho"],config,device) for spec in config["mixed_models"]];runs=pure+mixed
    first,held=map(str,config["evaluation_templates"])
    def val(run,status,layer,template,method,key="median_effect_fidelity"):return run["results"][status][str(layer)][template][method][key]
    r0=all(val(run,status,layer,template,"native")>=.9999 and val(run,status,layer,template,"native","final_answer_coordinate_r2")>=.9999 for run in runs for status in run["results"] for layer in config["layers"] for template in (first,held))
    r1=all(any(val(p,"all",layer,held,"pca_32")-val(m,"unselected",layer,held,"pca_32")>=.05 for layer in (1,2)) for p,m in zip(pure,mixed))
    r2=all(any(val(p,"all",layer,held,"answer_span")-val(m,"unselected",layer,held,"answer_span")>=.15 for layer in (1,2)) for p,m in zip(pure,mixed))
    r3=all(val(run,status,3,held,"answer_span")>=.90 for run in runs for status in run["results"])
    r4=all(sum(p["results"]["all"][str(layer)][held]["rank_k90"] for layer in (1,2))<sum(m["results"]["unselected"][str(layer)][held]["rank_k90"] for layer in (1,2)) for p,m in zip(pure,mixed))
    r5=all(abs(val(run,"selected",layer,held,"pca_64")-val(run,"unselected",layer,held,"pca_64"))<=.10 for run in mixed for layer in (2,3))
    r6=all(abs(val(run,status,layer,first,"pca_64")-val(run,status,layer,held,"pca_64"))<=.10 for run in runs for status in run["results"] for layer in config["layers"])
    r7=all(row["effectful_examples"]>=64 for run in runs for status in run["results"] for layer in config["layers"] for template in (first,held) for name,row in run["results"][status][str(layer)][template].items() if isinstance(row,dict) and "effectful_examples" in row)
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(r0,r1,r2,r3,r4,r5,r6,r7))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Pure-rule versus mixed-memory finite gate profiles","","Every cell uses a literal gate patch and exact suffix recomputation. Pure and mixed checkpoints share the data manifest but are independently trained; paired rows are comparison slots, not neuron alignments.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Regime | Seed | Status | Layer | PCA32/64 | answer span | K90 |","|---|---:|---|---:|---:|---:|---:|"]
    for run in runs:
        for status in run["results"]:
            for layer in config["layers"]:
                row=run["results"][status][str(layer)][held]
                lines.append(f"| {run['regime']} | {run['seed']} | {status} | {layer} | {row['pca_32']['median_effect_fidelity']:.3f}/{row['pca_64']['median_effect_fidelity']:.3f} | {row['answer_span']['median_effect_fidelity']:.3f} | {row['rank_k90']} |")
    lines += ["","A global mixed-regime difference means arbitrary load changes the shared implementation used even by unselected rule facts. It does not show that any individual unselected fact is memorized or that native neurons align across checkpoints."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
