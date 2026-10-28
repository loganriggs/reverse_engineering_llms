#!/usr/bin/env python3
"""Factorially attribute Layer-11 boundary contrasts to residual, embedding, and attention."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_late_address_program_audit import load_rows
from run_bilin12_layer5_weight_conditioned_addresses import derive_cleaned_addresses, output_conditioned_form
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layer11_boundary_upstream_decomposition.json")
OUTPUT=TENSOR_ROOT/"bilin12_layer11_boundary_upstream_decomposition.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYER11_BOUNDARY_UPSTREAM_DECOMPOSITION.md"
FIGURE=TENSOR_ROOT/"bilin12_layer11_boundary_upstream_decomposition.png"
FACTORS=("carried_residual","direct_embedding","attention")


def vector_shapley(outcomes):
    count=len(FACTORS); allocations={factor:torch.zeros_like(outcomes[0]) for factor in FACTORS}
    for factor_index,factor in enumerate(FACTORS):
        bit=1<<factor_index
        for mask in range(1<<count):
            if mask&bit: continue
            size=mask.bit_count(); weight=math.factorial(size)*math.factorial(count-size-1)/math.factorial(count)
            allocations[factor]+=weight*(outcomes[mask|bit]-outcomes[mask])
    return allocations


def fisher_transform(delta,probability):
    centered=delta-(probability*delta).sum(-1,keepdim=True)
    return probability.sqrt()*centered


@torch.inference_mode()
def prefix(model,ids,layer):
    dimension=model.config.n_embd; x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; value_bus=None
    for block in model.blocks[:layer]: x,value_bus=block(x,value_bus,x0)
    return x0,x,value_bus


def accumulate_metric(store,name,full,allocations,projection=None):
    if projection is not None:
        full=full@projection; allocations={key:value@projection for key,value in allocations.items()}
    denominator=float(full.square().sum())
    store[f"{name}/denominator"]+=denominator
    reconstructed=sum(allocations.values())
    store[f"{name}/reconstruction_error"]+=float((full-reconstructed).square().sum())
    for factor,value in allocations.items():
        store[f"{name}/attribution/{factor}"]+=float((value*full).sum())
        store[f"{name}/allocation_energy/{factor}"]+=float(value.square().sum())


@torch.inference_mode()
def evaluate(model,tokenizer,spec,config,contrast_basis):
    rows=load_rows(spec); newline=tokenizer.encode("\n",add_special_tokens=False)[0]; block=model.blocks[config["layer"]]; store=defaultdict(float); examples=0
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); original=torch.as_tensor(raw,device="cuda"); keep=original[:,config["position"]]!=newline
        if not bool(keep.any()): continue
        original=original[keep]; changed=original.clone(); changed[:,config["position"]]=newline; examples+=len(original)
        x0,x,bus=prefix(model,original,config["layer"]); x0c,xc,busc=prefix(model,changed,config["layer"])
        mixed=block.lambdas[0]*x+block.lambdas[1]*x0; mixedc=block.lambdas[0]*xc+block.lambdas[1]*x0c
        attention,_=block.attn(F.rms_norm(mixed,(model.config.n_embd,)),bus); attentionc,_=block.attn(F.rms_norm(mixedc,(model.config.n_embd,)),busc)
        position=config["position"]; pre=mixed[:,position]+attention[:,position]
        components={"carried_residual":block.lambdas[0]*(xc[:,position]-x[:,position]),"direct_embedding":block.lambdas[1]*(x0c[:,position]-x0[:,position]),"attention":attentionc[:,position]-attention[:,position]}
        outcomes={"state":{},"write":{},"logits":{}}
        for mask in range(8):
            candidate=pre.clone()
            for index,factor in enumerate(FACTORS):
                if mask&(1<<index): candidate+=components[factor]
            state=F.rms_norm(candidate,(model.config.n_embd,)); write=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias; raw_logits=model.lm_head(F.rms_norm(candidate+write,(model.config.n_embd,))).float(); logits=30*torch.tanh(raw_logits/30)
            outcomes["state"][mask]=state; outcomes["write"][mask]=write; outcomes["logits"][mask]=logits
        state_phi=vector_shapley(outcomes["state"]); write_phi=vector_shapley(outcomes["write"]); logit_phi=vector_shapley(outcomes["logits"])
        state_full=outcomes["state"][7]-outcomes["state"][0]; write_full=outcomes["write"][7]-outcomes["write"][0]; logit_full=outcomes["logits"][7]-outcomes["logits"][0]
        probability=F.softmax(outcomes["logits"][0],-1); fisher_full=fisher_transform(logit_full,probability); fisher_phi={key:fisher_transform(value,probability) for key,value in logit_phi.items()}
        accumulate_metric(store,"state",state_full,state_phi); accumulate_metric(store,"contrast",state_full,state_phi,contrast_basis); accumulate_metric(store,"write",write_full,write_phi); accumulate_metric(store,"fisher_logit",fisher_full,fisher_phi)
        for name,full,values in (("state",state_full,outcomes["state"]),("write",write_full,outcomes["write"]),("fisher_logit",fisher_full,{mask:fisher_transform(outcomes["logits"][mask]-outcomes["logits"][0],probability) for mask in range(8)})):
            additive=sum(values[1<<i]-values[0] for i in range(3)); store[f"{name}/additive_residual_num"]+=float((full-additive).square().sum()); store[f"{name}/additive_residual_den"]+=float(full.square().sum())
        pre_full=sum(components.values()); pre_den=float(pre_full.square().sum()); store["pre/denominator"]+=pre_den
        for factor,value in components.items(): store[f"pre/attribution/{factor}"]+=float((value*pre_full).sum())
    result={"examples":examples}
    for name in ("pre","state","contrast","write","fisher_logit"):
        denominator=store[f"{name}/denominator"]
        result[name]={"attribution":{factor:store[f"{name}/attribution/{factor}"]/max(denominator,1e-30) for factor in FACTORS}}
        if name!="pre":
            result[name]["reconstruction_relative_error"]=float(np.sqrt(store[f"{name}/reconstruction_error"]/max(denominator,1e-30)))
            result[name]["allocation_energy_ratio"]={factor:store[f"{name}/allocation_energy/{factor}"]/max(denominator,1e-30) for factor in FACTORS}
        if name in ("state","write","fisher_logit"):
            result[name]["additive_residual"]=float(np.sqrt(store[f"{name}/additive_residual_num"]/max(store[f"{name}/additive_residual_den"],1e-30)))
    return result


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    states,_=collect_states_writes(model,config["layer"],config["covariance_discovery"],config["covariance_positions"],config["batch_size"]); square,_,covariance=ridge_covariance_transforms(states,config["covariance_ridge_fraction"]); block=model.blocks[config["layer"]]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float(); _,outputs=output_map_eigensystem(down,left,right); form=output_conditioned_form(down,left,right,outputs[:,0]); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"]); u,_,_=torch.linalg.svd(edits,full_matrices=False); contrast_basis=u[:,2:4]
    results={corpus:evaluate(model,tokenizer,spec,config,contrast_basis) for corpus,spec in config["evaluation"].items()}
    dominant=[max(FACTORS,key=lambda f:results[c]["contrast"]["attribution"][f]) for c in results]
    hypotheses={
        "U1_one_path_dominates_boundary_contrast":{"supported":all(max(r["contrast"]["attribution"].values())>=.50 for r in results.values())},
        "U2_dominant_contrast_path_is_reproducible":{"supported":len(set(dominant))==1,"dominant_paths":dominant},
        "U3_direct_embedding_exceeds_attention_for_contrast":{"supported":all(r["contrast"]["attribution"]["direct_embedding"]>r["contrast"]["attribution"]["attention"] for r in results.values())},
        "U4_attention_is_material_for_final_logits":{"supported":all(r["fisher_logit"]["attribution"]["attention"]>=.10 for r in results.values())},
        "U5_state_contrast_and_mlp_write_have_similar_sources":{"supported":all(sum(abs(r["contrast"]["attribution"][f]-r["write"]["attribution"][f]) for f in FACTORS)<=.30 for r in results.values())},
        "U6_vector_shapley_is_exact":{"supported":all(r[name]["reconstruction_relative_error"]<=1e-5 for r in results.values() for name in ("state","contrast","write","fisher_logit"))},
        "U7_path_interactions_are_small_for_state":{"supported":all(r["state"]["additive_residual"]<=.10 for r in results.values())},
        "U8_path_interactions_are_material_for_mlp_write":{"supported":all(r["write"]["additive_residual"]>=.20 for r in results.values())},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"results":results},indent=2,allow_nan=False))

    corpora=list(results); metrics=("pre","contrast","write","fisher_logit"); fig,axes=plt.subplots(1,4,figsize=(18,4.5)); x=np.arange(len(corpora)); width=.22
    for axis,metric in zip(axes,metrics):
        for index,factor in enumerate(FACTORS): axis.bar(x+(index-1)*width,[results[c][metric]["attribution"][factor] for c in corpora],width,label=factor)
        axis.axhline(0,color="black",lw=.8); axis.set_xticks(x,corpora,rotation=20); axis.set_title(metric); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Upstream path decomposition of the Layer-11 boundary contrasts","","At the final block, the pre-MLP state is the sum of a carried residual term, a freshly injected normalized embedding, and attention output. For real final-token-to-newline edits, all eight factorial combinations of these three exact pre-state changes are evaluated. Vector Shapley allocations sum exactly to the changed normalized state, PC3–PC4 contrast, MLP write, and Fisher-centered final logits.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Upstream decomposition](bilin12_layer11_boundary_upstream_decomposition.png)","","| Corpus/metric | Carried | Embedding | Attention | Additive residual |","|---|---:|---:|---:|---:|"]
    for corpus in corpora:
        for metric in ("pre","state","contrast","write","fisher_logit"):
            row=results[corpus][metric]; residual=row.get("additive_residual",float("nan")); lines.append(f"| {corpus}/{metric} | {row['attribution']['carried_residual']:.3f} | {row['attribution']['direct_embedding']:.3f} | {row['attribution']['attention']:.3f} | {residual:.3f} |")
    lines += ["","Signed attributions are inner products between each exact vector-Shapley allocation and the full change, normalized so the three paths sum to one. Negative values indicate cancellation. The additive residual separately measures how wrong it would be to add the three factor-alone effects without their normalization and bilinear interactions."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
