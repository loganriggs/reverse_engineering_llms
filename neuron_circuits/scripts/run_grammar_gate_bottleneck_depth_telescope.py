#!/usr/bin/env python3
"""Literal finite causal dimension and semantic quotient across all MLP layers."""

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


CONFIG=Path("configs/grammar_gate_bottleneck_depth_telescope.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_gate_bottleneck_depth_telescope.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_GATE_BOTTLENECK_DEPTH_TELESCOPE.md")


@torch.inference_mode()
def native_data(model,prompts,layers,device,batch_size=512):
    answer_ids=list(VOCAB.answers);stores={"logits":[],"gates":[],"zero_logits":{layer:[] for layer in layers}}
    for start in range(0,len(prompts),batch_size):
        tokens=torch.as_tensor(prompts[start:start+batch_size].astype(np.int64),device=device)
        logits,acts=model(tokens,return_activations=True);stores["logits"].append(logits[:,-1,answer_ids].float().cpu())
        stores["gates"].append(torch.stack(acts,1).float().cpu())
        for layer in layers:
            zeros=torch.zeros_like(acts[layer]);candidate=model(tokens,patch=ActivationPatch(layer,zeros))
            stores["zero_logits"][layer].append(candidate[:,-1,answer_ids].float().cpu())
    return {"logits":torch.cat(stores["logits"]),"gates":torch.cat(stores["gates"]),
            "zero_logits":{layer:torch.cat(values) for layer,values in stores["zero_logits"].items()}}


@torch.inference_mode()
def patched_data(model,prompts,layer,values,device,batch_size=512):
    answer_ids=list(VOCAB.answers);logits=[];gate3=[]
    for start in range(0,len(prompts),batch_size):
        tokens=torch.as_tensor(prompts[start:start+batch_size].astype(np.int64),device=device);current=values[start:start+batch_size].to(device)
        output,acts=model(tokens,patch=ActivationPatch(layer,current),return_activations=True)
        logits.append(output[:,-1,answer_ids].float().cpu());gate3.append(acts[3].float().cpu())
    return torch.cat(logits),torch.cat(gate3)


def metric(candidate_logits,candidate_gate3,data,layer,labels,answer_mean,answer_basis,threshold):
    base=data["zero_logits"][layer];energy,cosine,fidelity=fisher_rows(base,data["logits"]-base,candidate_logits-base);keep=energy>=threshold
    target=(data["gates"][:,3]-answer_mean)@answer_basis.T;predicted=(candidate_gate3-answer_mean)@answer_basis.T
    r2=float(1-(predicted-target).square().sum()/(target-target.mean(0)).square().sum().clamp_min(1e-30))
    return {"effectful_examples":int(keep.sum()),"median_effect_fidelity":float(fidelity[keep].median()),"median_effect_cosine":float(cosine[keep].median()),
            "answer_accuracy":float((candidate_logits[keep].argmax(-1)==torch.as_tensor(labels)[keep]).float().mean()),"final_answer_coordinate_r2":r2}


def analyze(spec,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz");model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"]);output={}
    for flag,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[flag]["discovery"];eval_idx=panel[flag]["evaluation"]
        train=native_data(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),config["layers"],device)
        answer3=answer_table(train["gates"][:,3],labels[train_idx]);answer3_mean,answer3_basis,_=centroid_basis(answer3)
        fitted={}
        for layer in config["layers"]:
            values=train["gates"][:,layer];mean,basis=pca(values);down=model.blocks[layer].down.weight.detach().cpu()
            importance=values.var(0)*down.square().sum(0);answer=answer_table(values,labels[train_idx]);answer_mean,answer_basis,_=centroid_basis(answer)
            fitted[layer]={"mean":mean,"basis":basis,"coordinate_order":importance.argsort(descending=True),"answer_mean":answer_mean,"answer_basis":answer_basis}
        output[status]={str(layer):{} for layer in config["layers"]}
        for template in config["evaluation_templates"]:
            prompts=encode_prompts(manifest,facts[eval_idx],template);data=native_data(model,prompts,config["layers"],device)
            for layer in config["layers"]:
                fit=fitted[layer];native=data["gates"][:,layer];methods={}
                candidates={"native":native,"answer_span":project(native,fit["answer_mean"],fit["answer_basis"])}
                for rank in config["ranks"]:
                    candidates[f"pca_{rank}"]=project(native,fit["mean"],fit["basis"][:rank])
                    coordinate=fit["mean"].expand_as(native).clone();chosen=fit["coordinate_order"][:rank];coordinate[:,chosen]=native[:,chosen]
                    candidates[f"coordinate_{rank}"]=coordinate
                for name,candidate in candidates.items():
                    logits,gate3=patched_data(model,prompts,layer,candidate,device)
                    methods[name]=metric(logits,gate3,data,layer,labels[eval_idx],answer3_mean,answer3_basis,config["minimum_fisher_effect_norm"])
                methods["rank_k90"]=next((rank for rank in config["ranks"] if methods[f"pca_{rank}"]["median_effect_fidelity"]>=.90),193)
                generator=torch.Generator().manual_seed(config["seed"]+spec["seed"]*1000+layer*100+template*10+int(flag));random=[]
                for _ in range(config["random_subspaces"]):
                    basis=torch.linalg.qr(torch.randn(192,64,generator=generator)).Q.T;candidate=project(native,fit["mean"],basis)
                    logits,gate3=patched_data(model,prompts,layer,candidate,device)
                    random.append(metric(logits,gate3,data,layer,labels[eval_idx],answer3_mean,answer3_basis,config["minimum_fisher_effect_norm"])["median_effect_fidelity"])
                methods["random_64"]={"median_fidelity_mean":float(np.mean(random)),"median_fidelity_p95":float(np.quantile(random,.95)),"replicates":len(random)}
                output[status][str(layer)][str(template)]=methods
    return {"seed":spec["seed"],"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu";runs=[analyze(spec,config,device) for spec in config["models"]]
    first,held=map(str,config["evaluation_templates"])
    def val(run,status,layer,template,method,key="median_effect_fidelity"):return run["results"][status][str(layer)][template][method][key]
    d0=all(val(run,status,layer,template,"native")>=.9999 and val(run,status,layer,template,"native","final_answer_coordinate_r2")>=.9999 for run in runs for status in ("selected","unselected") for layer in config["layers"] for template in (first,held))
    d1=all(all(val(run,"selected",layer,held,"answer_span")<.60 for layer in (0,1,2)) and val(run,"selected",3,held,"answer_span")>.90 for run in runs)
    d2=all(val(run,"selected",layer,held,"pca_64")>=.80 for run in runs for layer in config["layers"])
    d3=all(val(run,"selected",layer,held,"pca_64")-val(run,"selected",layer,held,"coordinate_64")>=.15 for run in runs for layer in config["layers"])
    d4=True
    for run in runs:
        ranks=[run["results"]["selected"][str(layer)][held]["rank_k90"] for layer in config["layers"]]
        d4 &= all(a>=b for a,b in zip(ranks,ranks[1:])) and ranks[-1]<ranks[0]
    d5=all(any(val(run,"unselected",layer,held,"pca_32")-val(run,"selected",layer,held,"pca_32")>=.05 for layer in (0,1)) for run in runs)
    d6=all(val(run,status,layer,held,"pca_64")>=run["results"][status][str(layer)][held]["random_64"]["median_fidelity_p95"]+.15 for run in runs for status in ("selected","unselected") for layer in config["layers"])
    d7=all(abs(val(run,"selected",layer,first,"pca_64")-val(run,"selected",layer,held,"pca_64"))<=.10 for run in runs for layer in config["layers"])
    d8=all(row["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for layer in config["layers"] for template in (first,held) for name,row in run["results"][status][str(layer)][template].items() if isinstance(row,dict) and "effectful_examples" in row)
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(d0,d1,d2,d3,d4,d5,d6,d7,d8))}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Finite gate-bottleneck depth telescope","","Every layer uses the same assay: project its native post-product gate in a discovery-frozen affine subspace, patch that gate, and recompute the entire suffix. Effects are relative to literally zeroing that layer's gate. Final answer-coordinate R2 uses the frozen Layer-3 answer quotient.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | Layer | PCA r16/r32/r64 | coord64 | answer span | K90 | random64 p95 |","|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            for layer in config["layers"]:
                row=run["results"][status][str(layer)][held]
                lines.append(f"| {run['seed']} | {status} | {layer} | {row['pca_16']['median_effect_fidelity']:.3f}/{row['pca_32']['median_effect_fidelity']:.3f}/{row['pca_64']['median_effect_fidelity']:.3f} | {row['coordinate_64']['median_effect_fidelity']:.3f} | {row['answer_span']['median_effect_fidelity']:.3f} | {row['rank_k90']} | {row['random_64']['median_fidelity_p95']:.3f} |")
    lines += ["","This telescope distinguishes causal dimension from semantic organization. A layer can be compact without yet aligning to the answer quotient, or semantically aligned while using many native neurons."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
