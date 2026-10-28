#!/usr/bin/env python3
"""Exact upstream tensor-pair decomposition of the final answer quotient."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import prototypes, apply_prototypes, score, select_panel
from run_grammar_final_gate_code_dimension import eligible_mask
from run_grammar_final_gate_answer_quotient import answer_table, centroid_basis


CONFIG=Path("configs/grammar_answer_quotient_tensor_pairs.json")
OUTPUT=Path("artifacts/grammar_regimes/grammar_answer_quotient_tensor_pairs.json")
REPORT=Path("artifacts/grammar_regimes/GRAMMAR_ANSWER_QUOTIENT_TENSOR_PAIRS.md")
COMPONENTS=("embedding","b0_attn","b0_mlp","b1_attn","b1_mlp","b2_attn","b2_mlp","b3_attn")


def attention_write(block,hidden):
    normalized=block.attn_norm(hidden);q,k,v=block.qkv(normalized).chunk(3,-1)
    batch,length,width=hidden.shape
    def heads(x):return x.view(batch,length,block.heads,block.head_dim).transpose(1,2)
    attended=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),is_causal=True)
    return block.attn_out(attended.transpose(1,2).reshape_as(hidden))


@torch.inference_mode()
def collect(model,prompts,device,batch_size=128):
    stores={key:[] for key in ("pairs","gate","write","after","zero_logits","logits")}
    max_gate_error=max_write_error=max_forward_error=0.0;answer_ids=list(VOCAB.answers)
    for start in range(0,len(prompts),batch_size):
        tokens=torch.as_tensor(prompts[start:start+batch_size].astype(np.int64),device=device)
        hidden=model.embedding(tokens)+model.position[:tokens.shape[1]]
        components=[hidden[:,-1]]
        for block in model.blocks[:3]:
            attn=attention_write(block,hidden);hidden=hidden+attn;components.append(attn[:,-1])
            normalized=block.mlp_norm(hidden);activation=block.left(normalized)*block.right(normalized)
            mlp=activation@block.down.weight.T;hidden=hidden+mlp;components.append(mlp[:,-1])
        block=model.blocks[3];attn=attention_write(block,hidden);after=hidden+attn;components.append(attn[:,-1])
        after_final=after[:,-1];eps=block.mlp_norm.eps
        if eps is None:eps=torch.finfo(after_final.dtype).eps
        scale=block.mlp_norm.weight/torch.sqrt(after_final.square().mean(-1,keepdim=True)+eps)
        normalized_components=torch.stack(components,1)*scale[:,None,:]
        left=torch.einsum("bcw,hw->bch",normalized_components,block.left.weight)
        right=torch.einsum("bcw,hw->bch",normalized_components,block.right.weight)
        pairs=left[:,:,None,:]*right[:,None,:,:]
        gate=pairs.sum((1,2));native_gate=block.left(block.mlp_norm(after_final))*block.right(block.mlp_norm(after_final))
        write=gate@block.down.weight.T;native_write=native_gate@block.down.weight.T
        logits=model.unembedding(model.final_norm(after_final+write))[:,answer_ids].float()
        zero=model.unembedding(model.final_norm(after_final))[:,answer_ids].float()
        native_logits=model(tokens)[:,-1,answer_ids].float()
        max_gate_error=max(max_gate_error,float((gate-native_gate).abs().max()))
        max_write_error=max(max_write_error,float((write-native_write).abs().max()))
        max_forward_error=max(max_forward_error,float((logits-native_logits).abs().max()))
        for key,value in (("pairs",pairs),("gate",gate),("write",write),("after",after_final),("zero_logits",zero),("logits",logits)):
            stores[key].append(value.cpu())
    return {key:torch.cat(value) for key,value in stores.items()},{"maximum_gate_error":max_gate_error,"maximum_write_error":max_write_error,"maximum_forward_error":max_forward_error}


def origin_r2(predicted,target):
    return float(1-(predicted-target).square().sum()/target.square().sum().clamp_min(1e-30))


def analyze(spec,config,device):
    root=Path(spec["root"]);manifest=load_manifest(root/"manifest.npz")
    model=load_model(root/f"rho_{config['rho']:g}/depth_4/checkpoint.pt",device)
    facts=np.asarray(manifest["train_facts"]);labels,selected,changed=labels_for_rho(manifest,"train",config["rho"])
    eligible=eligible_mask(model,manifest,facts,labels,config["evaluation_templates"],device)
    panel=select_panel(labels,selected,changed,eligible,facts,config,config["seed"]+spec["seed"])
    down=model.blocks[3].down.weight.detach().cpu();output={};errors={"maximum_gate_error":0.0,"maximum_write_error":0.0,"maximum_forward_error":0.0}
    for flag,status in ((False,"unselected"),(True,"selected")):
        train_idx=panel[flag]["discovery"];eval_idx=panel[flag]["evaluation"]
        train,err=collect(model,encode_prompts(manifest,facts[train_idx],config["discovery_template"]),device)
        for key in errors:errors[key]=max(errors[key],err[key])
        pair_mean=train["pairs"].mean(0);gate_mean=pair_mean.sum((0,1))
        _,answer_basis,_=centroid_basis(answer_table(train["gate"],labels[train_idx]))
        centered_train=train["pairs"]-pair_mean
        coordinates=torch.einsum("bijh,rh->bijr",centered_train,answer_basis)
        energy=coordinates.square().mean((0,3));order=energy.flatten().argsort(descending=True)
        write_table=prototypes(train["write"],facts[train_idx,2],labels[train_idx],True)
        output[status]={"pair_order":[{"left":COMPONENTS[int(i)//8],"right":COMPONENTS[int(i)%8],"energy":float(energy.flatten()[i])} for i in order]}
        for template in config["evaluation_templates"]:
            data,err=collect(model,encode_prompts(manifest,facts[eval_idx],template),device)
            for key in errors:errors[key]=max(errors[key],err[key])
            centered=data["pairs"]-pair_mean;flat=centered.flatten(1,2)
            target_coordinates=(data["gate"]-gate_mean)@answer_basis.T
            oracle=apply_prototypes(write_table,facts[eval_idx,2],labels[eval_idx],True)
            def assess(mask):
                chosen=torch.as_tensor(mask,dtype=torch.long);candidate=gate_mean+flat[:,chosen].sum(1)
                row=score(candidate@down.T,data["write"],data["after"],data["zero_logits"],data["logits"],labels[eval_idx],oracle,model,config["minimum_fisher_effect_norm"])
                row["answer_coordinate_r2"]=origin_r2((candidate-gate_mean)@answer_basis.T,target_coordinates)
                return row
            methods={}
            for count in config["pair_counts"]:methods[f"top_{count}"]=assess(order[:count])
            masks={
                "self_only":[i*8+i for i in range(8)],
                "cross_only":[i*8+j for i in range(8) for j in range(8) if i!=j],
                "upstream_7x7":[i*8+j for i in range(7) for j in range(7)],
                "final_attention_related":[i*8+j for i in range(8) for j in range(8) if i==7 or j==7],
            }
            for name,mask in masks.items():methods[name]=assess(mask)
            frontier=[]
            for count in range(1,65):
                candidate=gate_mean+flat[:,order[:count]].sum(1)
                frontier.append(origin_r2((candidate-gate_mean)@answer_basis.T,target_coordinates))
            methods["coordinate_k90"]=next((i+1 for i,value in enumerate(frontier) if value>=.90),65)
            methods["coordinate_frontier"]=frontier
            rng=np.random.default_rng(config["seed"]+spec["seed"]*100+template+int(flag));random=[]
            for _ in range(config["random_subsets"]):random.append(assess(rng.choice(64,16,replace=False).tolist())["median_effect_fidelity"])
            methods["random_16"]={"median_fidelity_mean":float(np.mean(random)),"median_fidelity_p95":float(np.quantile(random,.95)),"replicates":len(random)}
            output[status][str(template)]=methods
    return {"seed":spec["seed"],"errors":errors,"results":output}


def main():
    config=json.loads(CONFIG.read_text());device="cuda" if torch.cuda.is_available() else "cpu";runs=[analyze(spec,config,device) for spec in config["models"]]
    first,held=map(str,config["evaluation_templates"])
    def val(run,status,template,method,key="median_effect_fidelity"):return run["results"][status][template][method][key]
    t0=all(max(run["errors"].values())<1e-4 and val(run,status,template,"top_64")>=.9999 for run in runs for status in ("selected","unselected") for template in (first,held))
    t1=all(val(run,"selected",held,"top_16")>=.80 for run in runs)
    t2=all(val(run,"selected",held,"top_8")<.80 for run in runs)
    t3=all(val(run,status,held,"top_16")>=run["results"][status][held]["random_16"]["median_fidelity_p95"]+.15 for run in runs for status in ("selected","unselected"))
    t4=all(val(run,"selected",held,"self_only")<.60 for run in runs)
    t5=all(val(run,"selected",held,"upstream_7x7")>=.75 for run in runs)
    t6=all(run["results"]["selected"][held]["coordinate_k90"]>=run["results"]["unselected"][held]["coordinate_k90"]+4 for run in runs)
    t7=all(val(run,"selected",held,"top_16","answer_coordinate_r2")>=.70 for run in runs)
    t8=all(abs(val(run,"selected",first,"top_16")-val(run,"selected",held,"top_16"))<=.10 for run in runs)
    t9=all(row["effectful_examples"]>=64 for run in runs for status in ("selected","unselected") for template in (first,held) for name,row in run["results"][status][template].items() if isinstance(row,dict) and "effectful_examples" in row)
    hypotheses={name:{"supported":bool(value)} for name,value in zip(config["hypotheses"],(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9))}
    OUTPUT.write_text(json.dumps({"config":config,"components":COMPONENTS,"hypotheses":hypotheses,"runs":runs},indent=2,allow_nan=False))
    lines=["# Exact upstream tensor pairs for the final answer quotient","","The final normalized MLP input is decomposed into eight additive residual components using one shared RMS scale. The bilinear gate is therefore the exact sum of 64 ordered left-component × right-component terms. Discovery pair means are retained; ranked subsets preserve selected pair variations and mean-fill the rest.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Seed | Status | top 4/8/16/32 | self/cross | upstream7x7/final-attn | coord K90 | random16 mean/p95 |","|---:|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        for status in ("unselected","selected"):
            row=run["results"][status][held];random=row["random_16"]
            lines.append(f"| {run['seed']} | {status} | {row['top_4']['median_effect_fidelity']:.3f}/{row['top_8']['median_effect_fidelity']:.3f}/{row['top_16']['median_effect_fidelity']:.3f}/{row['top_32']['median_effect_fidelity']:.3f} | {row['self_only']['median_effect_fidelity']:.3f}/{row['cross_only']['median_effect_fidelity']:.3f} | {row['upstream_7x7']['median_effect_fidelity']:.3f}/{row['final_attention_related']['median_effect_fidelity']:.3f} | {row['coordinate_k90']} | {random['median_fidelity_mean']:.3f}/{random['median_fidelity_p95']:.3f} |")
    lines += ["","This is an exact affine variation decomposition of one tensor contraction. A compact pair frontier shortens the upstream-to-gate description but does not yet identify which token interactions each residual component computed. A broad frontier would not falsify the seven-dimensional output quotient; it would locate complexity in constructing its coordinates."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",OUTPUT,REPORT,flush=True)


if __name__=="__main__":main()
