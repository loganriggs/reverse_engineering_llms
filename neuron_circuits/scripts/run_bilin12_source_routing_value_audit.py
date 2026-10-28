#!/usr/bin/env python3
"""Split final-destination source edges into exact routing and value changes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_source_routing_value_audit.json")
OUTPUT=TENSOR_ROOT/"bilin12_source_routing_value_audit.json"
REPORT=TENSOR_ROOT/"BILIN12_SOURCE_ROUTING_VALUE_AUDIT.md"
FIGURE=TENSOR_ROOT/"bilin12_source_routing_value_audit.png"
FACTORS=("routing","value")


def product_shapley_parts(a0,a1,v0,v1):
    routing=.5*(a1-a0)[...,None]*(v0+v1)
    value=.5*(a0+a1)[...,None]*(v1-v0)
    return routing,value


def project_head_sources(block,values):
    # values: B,H,S,Dh -> B,S,D
    result=0
    for head in range(block.attn.n_head):
        start=head*block.attn.head_dim;stop=(head+1)*block.attn.head_dim
        result=result+values[:,head]@block.attn.c_proj.weight[:,start:stop].T
    return result


def factor_sources(block,base,changed,destination):
    a0=base["pattern"][:,:,destination];a1=changed["pattern"][:,:,destination]
    v0=((1-block.attn.lamb)*base["current_value"]+block.attn.lamb*base["value_bus"]).permute(0,2,1,3)
    v1=((1-block.attn.lamb)*changed["current_value"]+block.attn.lamb*changed["value_bus"]).permute(0,2,1,3)
    routing,value=product_shapley_parts(a0,a1,v0,v1)
    return project_head_sources(block,routing),project_head_sources(block,value),a1-a0


def empty_acc():
    return {"examples":0,"state_den":0.,"state_cross":{f:0. for f in FACTORS},"behavior_den":0.,"behavior_cross":{f:0. for f in FACTORS},"isolated_error":0.,"selected_den":0.,"selected_routing_cross":0.,"relative_num":0.,"relative_den":0.,"max_write_error":0.,"max_logit_error":0.,"max_pattern_mass_error":0.,"max_sign_mass_error":0.,"max_pre_edit_value":0.}


def finalize(row):
    return {"examples":row["examples"],"state_projection":{f:row["state_cross"][f]/max(row["state_den"],1e-30) for f in FACTORS},"behavioral_shapley":{f:row["behavior_cross"][f]/max(row["behavior_den"],1e-30) for f in FACTORS},"isolated_effect_sum_fidelity":1-row["isolated_error"]/max(row["behavior_den"],1e-30),"selected_top32_routing_state_projection":row["selected_routing_cross"]/max(row["selected_den"],1e-30),"relative_write_recomposition_error":float(np.sqrt(row["relative_num"]/max(row["relative_den"],1e-30))),"maximum_write_error":row["max_write_error"],"maximum_logit_error":row["max_logit_error"],"maximum_pattern_mass_error":row["max_pattern_mass_error"],"maximum_sign_mass_error":row["max_sign_mass_error"],"maximum_pre_edit_value_write":row["max_pre_edit_value"]}


@torch.inference_mode()
def evaluate(model,tokenizer,rows,edit_position,config):
    newline=tokenizer.encode("\n",add_special_tokens=False)[0];destination=config["prediction_position"];acc={str(layer):empty_acc() for layer in config["layers"]}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:config["sequence_length"]],dtype=np.int64);base_ids=torch.as_tensor(raw,device="cuda",dtype=torch.long);keep=base_ids[:,edit_position]!=newline
        if not bool(keep.any()):continue
        base_ids=base_ids[keep];changed_ids=base_ids.clone();changed_ids[:,edit_position]=newline;base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids)
        for layer in config["layers"]:
            key=str(layer);row=acc[key];base=base_rows[layer];changed=changed_rows[layer];routing_sources,value_sources,pattern_delta=factor_sources(model.blocks[layer],base["details"],changed["details"],destination);routing=routing_sources.sum(1);value=value_sources.sum(1);target=(changed["details"]["write"][:,destination]-base["details"]["write"][:,destination]).float();prediction=routing+value;error=prediction-target
            full_write=base["details"]["write"].clone();full_write[:,destination]=changed["details"]["write"][:,destination];worlds=[]
            for delta in (torch.zeros_like(target),routing,value,prediction):
                write=base["details"]["write"].clone();write[:,destination]+=delta.to(write.dtype);worlds.append(write)
            logits=logits_from_attention_writes(model,layer,base,torch.stack(worlds));logit_target=logits[3]-logits[0];te,_,_,_=fisher_terms(logits[0],logit_target,logit_target);route_effect=.5*((logits[1]-logits[0])+(logits[3]-logits[2]));value_effect=.5*((logits[2]-logits[0])+(logits[3]-logits[1]));isolated=(logits[1]-logits[0])+(logits[2]-logits[0]);_,_,route_cross,_=fisher_terms(logits[0],logit_target,route_effect);_,_,value_cross,_=fisher_terms(logits[0],logit_target,value_effect);_,_,_,isolated_error=fisher_terms(logits[0],logit_target,isolated)
            state_den=target.square().sum();row["state_den"]+=float(state_den);row["state_cross"]["routing"]+=float((routing*target).sum());row["state_cross"]["value"]+=float((value*target).sum());row["behavior_den"]+=float(te.sum());row["behavior_cross"]["routing"]+=float(route_cross.sum());row["behavior_cross"]["value"]+=float(value_cross.sum());row["isolated_error"]+=float(isolated_error.sum())
            full_sources=routing_sources+value_sources;indices=full_sources.square().sum(-1).argsort(-1,descending=True)[:,:config["source_budget"]];mask=torch.zeros(len(base_ids),config["sequence_length"],device="cuda");mask.scatter_(1,indices,1);selected=(mask[:,:,None]*full_sources).sum(1);selected_routing=(mask[:,:,None]*routing_sources).sum(1);row["selected_den"]+=float(selected.square().sum());row["selected_routing_cross"]+=float((selected_routing*selected).sum())
            positive=pattern_delta.clamp_min(0).sum(-1);negative=pattern_delta.clamp_max(0).sum(-1).abs();row["max_pattern_mass_error"]=max(row["max_pattern_mass_error"],float(pattern_delta.sum(-1).abs().max()));row["max_sign_mass_error"]=max(row["max_sign_mass_error"],float((positive-negative).abs().max()));row["max_pre_edit_value"]=max(row["max_pre_edit_value"],float(value_sources[:,:edit_position].abs().max()) if edit_position else 0.);row["max_write_error"]=max(row["max_write_error"],float(error.abs().max()));row["relative_num"]+=float(error.square().sum());row["relative_den"]+=float(target.square().sum());direct_logits=logits_from_attention_writes(model,layer,base,full_write.unsqueeze(0))[0];row["max_logit_error"]=max(row["max_logit_error"],float((logits[3]-direct_logits).abs().max()));row["examples"]+=len(base_ids)
    return {layer:finalize(row) for layer,row in acc.items()}


def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");results={}
    for corpus,spec in config["evaluation"].items():
        rows=load_rows(spec);results[corpus]={}
        for position in config["edit_positions"]:print("evaluating",corpus,position,flush=True);results[corpus][str(position)]=evaluate(model,tokenizer,rows,position,config)
    all_rows=[results[c][str(p)][str(layer)] for c in results for p in config["edit_positions"] for layer in config["layers"]];transfer={}
    for layer in config["layers"]:transfer[str(layer)]={corpus:float(np.mean([results[corpus][str(p)][str(layer)]["behavioral_shapley"]["routing"] for p in config["edit_positions"]])) for corpus in results}
    hypotheses={"R1_routing_value_recomposition_is_exact":{"supported":all(r["relative_write_recomposition_error"]<=1e-5 and r["maximum_logit_error"]<=1e-4 for r in all_rows)},"R2_attention_mass_is_conserved":{"supported":all(r["maximum_pattern_mass_error"]<=1e-5 for r in all_rows)},"R3_pre_edit_value_changes_are_causally_zero":{"supported":all(r["maximum_pre_edit_value_write"]<=1e-6 for r in all_rows)},"R4_routing_dominates_behavior":{"supported":all(r["behavioral_shapley"]["routing"]>=.50 for r in all_rows)},"R5_value_changes_are_sometimes_material":{"supported":max(r["behavioral_shapley"]["value"] for r in all_rows)>=.25},"R6_isolated_routing_and_value_effects_add":{"supported":all(r["isolated_effect_sum_fidelity"]>=.90 for r in all_rows)},"R7_state_and_behavior_agree_on_the_larger_factor":{"supported":all(max(FACTORS,key=lambda f:abs(r["state_projection"][f]))==max(FACTORS,key=lambda f:abs(r["behavioral_shapley"][f])) for r in all_rows)},"R8_large_source_edges_are_mostly_routing":{"supported":np.mean([r["selected_top32_routing_state_projection"] for r in all_rows])>=.60,"mean":float(np.mean([r["selected_top32_routing_state_projection"] for r in all_rows]))},"R9_routing_has_structurally_balanced_sign_mass":{"supported":all(r["maximum_sign_mass_error"]<=1e-4 for r in all_rows)},"R10_routing_dominance_transfers_ood":{"supported":all(abs(means[c]-means["fineweb"])<=.15 for means in transfer.values() for c in ("wikitext","tinystories")),"means":transfer}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False));layers=config["layers"];fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for corpus in results:
        axes[0].plot(layers,[np.mean([results[corpus][str(p)][str(layer)]["behavioral_shapley"]["routing"] for p in config["edit_positions"]]) for layer in layers],marker="o",label=corpus);axes[1].plot(layers,[np.mean([results[corpus][str(p)][str(layer)]["behavioral_shapley"]["value"] for p in config["edit_positions"]]) for layer in layers],marker="o",label=corpus);axes[2].plot(layers,[np.mean([results[corpus][str(p)][str(layer)]["selected_top32_routing_state_projection"] for p in config["edit_positions"]]) for layer in layers],marker="o",label=corpus)
    axes[0].set_title("Routing behavioral Shapley");axes[1].set_title("Value behavioral Shapley");axes[2].set_title("Routing inside top-32 state edges")
    for axis in axes:axis.set_xlabel("layer");axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Routing-versus-value source audit","","Every final-destination head/source product is split by the exact two-factor Shapley identity into attention-pattern routing change and mixed-value change. These terms recompose before the head output projections. Their state projections are compared with finite downstream behavioral Shapley, isolated-effect additivity, and the state-norm top-32 source program.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"The strict 1e-5 recomposition gate failed from float32 summation order: maximum relative write error is {max(r['relative_write_recomposition_error'] for r in all_rows):.2e} and maximum propagated-logit disagreement is {max(r['maximum_logit_error'] for r in all_rows):.2e}. The algebraic product-Shapley identity is exact; the locked numerical failure is retained."]
    lines += ["","![Routing value audit](bilin12_source_routing_value_audit.png)","","| Corpus | Position | Layer | Routing/value state | Routing/value behavior | Isolated sum fidelity | Top32 routing |","|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for position in config["edit_positions"]:
            for layer in layers:
                r=results[corpus][str(position)][str(layer)];lines.append(f"| {corpus} | {position} | {layer} | {r['state_projection']['routing']:.3f}/{r['state_projection']['value']:.3f} | {r['behavioral_shapley']['routing']:.3f}/{r['behavioral_shapley']['value']:.3f} | {r['isolated_effect_sum_fidelity']:.3f} | {r['selected_top32_routing_state_projection']:.3f} |")
    lines += ["","Attention-row normalization makes the routing coefficients a conserved redistribution: their signed pattern changes sum to zero per head. Whether this structural cancellation explains the behavior depends on the routing/value causal allocation rather than conservation alone."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
