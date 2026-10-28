#!/usr/bin/env python3
"""Finite causal bottleneck test inside the recurring Block-2 MLP hub."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import fisher_rows, select_panel
from run_grammar_final_gate_code_dimension import eligible_mask, pca, project
from run_grammar_final_gate_answer_quotient import answer_table, centroid_basis


CONFIG=Path("configs/grammar_block2_gate_bottleneck.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_block2_gate_bottleneck.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_BLOCK2_GATE_BOTTLENECK.md")


@torch.inference_mode()
def native_data(model,prompts,device,batch_size=512):
    answer_ids=list(VOCAB.answers);stores={key:[] for key in ("logits","zero_logits","gate2","gate3")}
    for start in range(0,len(prompts),batch_size):
        tokens=torch.as_tensor(prompts[start:start+batch_size].astype(np.int64),device=device)
        logits,acts=model(tokens,return_activations=True)
        zeros=torch.zeros_like(acts[2]);zero_logits=model(tokens,patch=ActivationPatch(2,zeros))
        for key,value in (("logits",logits[:,-1,answer_ids]),("zero_logits",zero_logits[:,-1,answer_ids]),("gate2",acts[2]),("gate3",acts[3])):
            stores[key].append(value.float().cpu())
    return {key:torch.cat(value) for key,value in stores.items()}


@torch.inference_mode()
def patched_data(model,prompts,values,device,batch_size=512):
    answer_ids=list(VOCAB.answers);logits=[];gate3=[]
    for start in range(0,len(prompts),batch_size):
        tokens=torch.as_tensor(prompts[start:start+batch_size].astype(np.int64),device=device)
        current=values[start:start+batch_size].to(device)
        output,acts=model(tokens,patch=ActivationPatch(2,current),return_activations=True)
        logits.append(output[:,-1,answer_ids].float().cpu());gate3.append(acts[3].float().cpu())
    return torch.cat(logits),torch.cat(gate3)


def origin_r2(predicted,target,mean):
    return float(1-(predicted-target).square().sum()/(target-mean).square().sum().clamp_min(1e-30))


def metrics(candidate_logits,candidate_gate3,data,labels,answer_mean,answer_basis,threshold):
    energy,cosine,fidelity=fisher_rows(data["zero_logits"],data["logits"]-data["zero_logits"],candidate_logits-data["zero_logits"])
    keep=energy>=threshold;target_coordinates=(data["gate3"]-answer_mean)@answer_basis.T
    candidate_coordinates=(candidate_gate3-answer_mean)@answer_basis.T
    return {"effectful_examples":int(keep.sum()),"median_effect_cosine":float(cosine[keep].median()),
            "median_effect_fidelity":float(fidelity[keep].median()),
            "answer_accuracy":float((candidate_logits[keep].argmax(-1)==torch.as_tensor(labels)[keep]).float().mean()),
            "final_answer_coordinate_r2":origin_r2(candidate_coordinates,target_coordinates,target_coordinates.mean(0))}


def analyze(spec,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"]);output={}
    down2=model.blocks[2].down.weight.detach().cpu()
    for flag,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[flag]["discovery"];eval_idx=panel[flag]["evaluation"]
        train_prompts=encode_prompts(manifest,facts[train_idx],config["discovery_template"]);train=native_data(model,train_prompts,device)
        mean,basis=pca(train["gate2"]);importance=train["gate2"].var(0)*down2.square().sum(0);coordinate_order=importance.argsort(descending=True)
        answer2=answer_table(train["gate2"],labels[train_idx]);answer2_mean,answer2_basis,_=centroid_basis(answer2)
        answer3=answer_table(train["gate3"],labels[train_idx]);answer3_mean,answer3_basis,_=centroid_basis(answer3)
        output[status]={}
        for template in config["evaluation_templates"]:
            prompts=encode_prompts(manifest,facts[eval_idx],template);data=native_data(model,prompts,device);methods={}
            candidates={}
            for rank in config["ranks"]:
                dense=mean.expand_as(data["gate2"]) if rank==0 else project(data["gate2"],mean,basis[:rank])
                coordinate=mean.expand_as(data["gate2"]).clone()
                if rank:coordinate[:,coordinate_order[:rank]]=data["gate2"][:,coordinate_order[:rank]]
                candidates[f"pca_{rank}"]=dense;candidates[f"coordinate_{rank}"]=coordinate
            candidates["native"]=data["gate2"]
            candidates["answer_span"]=project(data["gate2"],answer2_mean,answer2_basis)
            for name,candidate in candidates.items():
                logits,gate3=patched_data(model,prompts,candidate,device)
                methods[name]=metrics(logits,gate3,data,labels[eval_idx],answer3_mean,answer3_basis,config["minimum_fisher_effect_norm"])
            generator=torch.Generator().manual_seed(config["seed"]+spec["seed"]*100+template+int(flag));random=[]
            for _ in range(config["random_subspaces"]):
                random_basis=torch.linalg.qr(torch.randn(192,64,generator=generator)).Q.T
                candidate=project(data["gate2"],mean,random_basis);logits,gate3=patched_data(model,prompts,candidate,device)
                random.append(metrics(logits,gate3,data,labels[eval_idx],answer3_mean,answer3_basis,config["minimum_fisher_effect_norm"])["median_effect_fidelity"])
            methods["random_64"]={"median_fidelity_mean":float(np.mean(random)),"median_fidelity_p95":float(np.quantile(random,.95)),"replicates":len(random)}
            output[status][str(template)]=methods
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu";runs=[analyze(spec,config,device) for spec in config["models"]]
    first,held=map(str,config["evaluation_templates"])
    def val(run,status,template,method,key="median_effect_fidelity"):return run["results"][status][template][method][key]
    b0=all(val(run,status,template,"native")>=.9999 and val(run,status,template,"native","final_answer_coordinate_r2")>=.9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    b1=all(val(run,"selected",held,"pca_64")>=.80 for run in runs)
    b2=all(val(run,"selected",held,"pca_16")<.80 for run in runs)
    b3=all(val(run,"selected",held,"pca_64")-val(run,"selected",held,"coordinate_64")>=.15 for run in runs)
    b4=all(val(run,status,held,"pca_64")>=run["results"][status][held]["random_64"]["median_fidelity_p95"]+.15 for run in runs for status in ("selected","unselected"))
    b5=all(val(run,"unselected",held,"pca_32")-val(run,"selected",held,"pca_32")>=.05 for run in runs)
    b6=all(val(run,"selected",held,"answer_span")<.60 for run in runs)
    b7=all(val(run,"selected",held,"pca_64","final_answer_coordinate_r2")>=.70 for run in runs)
    b8=all(abs(val(run,"selected",first,"pca_64")-val(run,"selected",held,"pca_64"))<=.10 for run in runs)
    b9=all(row["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for template in (first,held) for name,row in run["results"][status][template].items() if isinstance(row,dict) and "effectful_examples" in row)
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Finite bottleneck inside the Block-2 MLP hub","","Block-2 post-product gates are projected in a discovery-frozen affine subspace, patched at the real model site, and Blocks 3 plus the complete output suffix are recomputed. The target effect is the native Block-2 MLP contribution relative to a literal zero-gate intervention.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | PCA r0/r16/r32/r64/r96 | coordinate64 | answer-span | random64 mean/p95 | final-coordinate R2@64 |","|---:|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            row=run["results"][status][held];random=row["random_64"]
            ranks="/".join(f"{row[f'pca_{r}']['median_effect_fidelity']:.3f}" for r in (0,16,32,64,96))
            lines.append(f"| {run['seed']} | {status} | {ranks} | {row['coordinate_64']['median_effect_fidelity']:.3f} | {row['answer_span']['median_effect_fidelity']:.3f} | {random['median_fidelity_mean']:.3f}/{random['median_fidelity_p95']:.3f} | {row['pca_64']['final_answer_coordinate_r2']:.3f} |")
    lines += ["","This is a literal finite replacement, unlike the affine final-pair decomposition. A compact dense Block-2 gate would identify an earlier causal bottleneck; a broad or status-specific result would localize arbitrary-address complexity inside this hub."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
