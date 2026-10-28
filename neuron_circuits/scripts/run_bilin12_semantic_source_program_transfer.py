#!/usr/bin/env python3
"""Transfer source-edge program rules from newline edits to semantic substitutions."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_source_fisher_gram_program import cooperative_order
from run_bilin12_source_routing_value_audit import factor_sources
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_semantic_source_program_transfer.json")
OUTPUT=TENSOR_ROOT/"bilin12_semantic_source_program_transfer.json"
REPORT=TENSOR_ROOT/"BILIN12_SEMANTIC_SOURCE_PROGRAM_TRANSFER.md"
FIGURE=TENSOR_ROOT/"bilin12_semantic_source_program_transfer.png"


def select_windows(rows,token,edit_position,prediction_position,count,rng):
    maximum_source=rows.shape[1]-(prediction_position-edit_position)-1;locations=np.argwhere(np.asarray(rows[:,edit_position:maximum_source+1])==token);rng.shuffle(locations);selected=[];used=set()
    for row,relative in locations:
        row=int(row);source=int(relative+edit_position)
        if row in used:continue
        selected.append((row,source-edit_position));used.add(row)
        if len(selected)==count:break
    if len(selected)<count:raise RuntimeError(f"token {token} has only {len(selected)} windows")
    return selected


def empty_acc():return {"target":0.,"error":0.,"examples":0}
def update_acc(row,target,error):row["target"]+=float(target);row["error"]+=float(error);row["examples"]+=1
def finish(row):return {"examples":row["examples"],"fisher_fidelity":1-row["error"]/max(row["target"],1e-30),"target_energy":row["target"]}


@torch.inference_mode()
def analyze_example(model,base_ids,changed_ids,config):
    base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids);results={}
    for layer in config["layers"]:
        base=base_rows[layer];changed=changed_rows[layer];block=model.blocks[layer];position=config["prediction_position"]
        base_sources=source_write_contributions(block,base["details"],position);changed_sources=source_write_contributions(block,changed["details"],position);source_delta=(changed_sources-base_sources).float();routing_sources,value_sources,pattern_delta=factor_sources(block,base["details"],changed["details"],position)
        full_write=base["details"]["write"].clone();full_write[:,position]=changed["details"]["write"][:,position];base_logits,target_logits=logits_from_attention_writes(model,layer,base,torch.stack((base["details"]["write"],full_write)));target_delta=target_logits-base_logits;probability=F.softmax(base_logits.float(),-1)
        edge_logits=[]
        for start in range(0,config["sequence_length"],config["world_chunk"]):
            worlds=[]
            for source in range(start,min(start+config["world_chunk"],config["sequence_length"])):
                write=base["details"]["write"].clone();write[:,position]+=source_delta[:,source].to(write.dtype);worlds.append(write)
            edge_logits.append(logits_from_attention_writes(model,layer,base,torch.stack(worlds)))
        edge_effects=(torch.cat(edge_logits)-base_logits[None]).permute(1,0,2).contiguous();oracle,_,_=cooperative_order(probability,target_delta,edge_effects)
        orders={"cooperative_oracle":oracle,"state_norm":source_delta.square().sum(-1).argsort(-1,descending=True),"pattern_change":pattern_delta.abs().sum(1).argsort(-1,descending=True),"routing_norm":routing_sources.square().sum(-1).argsort(-1,descending=True),"value_norm":value_sources.square().sum(-1).argsort(-1,descending=True),"recency":torch.arange(config["sequence_length"]-1,-1,-1,device="cuda")[None]}
        worlds=[];labels=[]
        for method in config["methods"]:
            for budget in config["budgets"]:
                indices=orders[method][:,:budget];mask=torch.zeros(1,config["sequence_length"],device="cuda");mask.scatter_(1,indices,1);approx=(mask[:,:,None]*source_delta).sum(1);write=base["details"]["write"].clone();write[:,position]+=approx.to(write.dtype);worlds.append(write);labels.append(f"{method}/{budget}")
        logits=logits_from_attention_writes(model,layer,base,torch.stack(worlds));target_energy,_,_,_=fisher_terms(base_logits,target_delta,target_delta);metrics={}
        for index,label in enumerate(labels):
            _,_,_,error=fisher_terms(base_logits,target_delta,logits[index]-base_logits);metrics[label]={"target_energy":float(target_energy.sum()),"error":float(error.sum()),"fisher_fidelity":float(1-error.sum()/target_energy.sum().clamp_min(1e-30))}
        results[str(layer)]={"full_kl":float(exact_kl(base_logits,target_logits).sum()),"metrics":metrics,"oracle_top16":oracle[0,:16].cpu().tolist(),"state_top16":orders["state_norm"][0,:16].cpu().tolist()}
    return results


def aggregate(records,config,selector=lambda record:True):
    rows={str(layer):{f"{method}/{budget}":empty_acc() for method in config["methods"] for budget in config["budgets"]} for layer in config["layers"]}
    for record in records:
        if not selector(record):continue
        for layer,result in record["layers"].items():
            for label,metric in result["metrics"].items():update_acc(rows[layer][label],metric["target_energy"],metric["error"])
    return {layer:{label:finish(row) for label,row in methods.items()} for layer,methods in rows.items()}


def weighted_over_layers(summary,label):
    target=sum(row[label]["target_energy"] for row in summary.values());error=sum((1-row[label]["fisher_fidelity"])*row[label]["target_energy"] for row in summary.values());return 1-error/max(target,1e-30)


def jaccard(left,right):
    a=set(left);b=set(right);return len(a&b)/max(len(a|b),1)


def main():
    config=json.loads(CONFIG.read_text());semantic=json.loads(Path(config["semantic_source"]).read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");records=[]
    for corpus_index,(corpus,spec) in enumerate(config["evaluation"].items()):
        source=np.load(spec["path"],mmap_mode="r");rows=source[spec["start"]:spec["start"]+spec["rows"]];rng=np.random.default_rng(config["seed"]+corpus_index)
        for family,pairs in semantic["families"].items():
            for source_token,target_token in pairs:
                for row_index,start in select_windows(rows,source_token,config["edit_position"],config["prediction_position"],config["contexts_per_pair"],rng):
                    base=np.asarray(rows[row_index,start:start+config["sequence_length"]],dtype=np.int64).copy();changed=base.copy();changed[config["edit_position"]]=target_token;layers=analyze_example(model,torch.as_tensor(base,device="cuda")[None],torch.as_tensor(changed,device="cuda")[None],config);records.append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","source_token":tokenizer.decode([source_token]),"target_token":tokenizer.decode([target_token]),"row":row_index,"start":start,"layers":layers});print(corpus,family,tokenizer.decode([source_token]),"->",tokenizer.decode([target_token]),flush=True)
    overall=aggregate(records,config);by_corpus={c:aggregate(records,config,lambda r,c=c:r["corpus"]==c) for c in config["evaluation"]};by_family={f:aggregate(records,config,lambda r,f=f:r["family"]==f) for f in semantic["families"]};by_corpus_family={c:{f:aggregate(records,config,lambda r,c=c,f=f:r["corpus"]==c and r["family"]==f) for f in semantic["families"]} for c in config["evaluation"]}
    def overall_fidelity(method,budget):return weighted_over_layers(overall,f"{method}/{budget}")
    family_fidelity={f:{m:weighted_over_layers(row,f"{m}/32") for m in config["methods"]} for f,row in by_family.items()}
    corpus_kl={c:float(np.median([layer["full_kl"] for r in records if r["corpus"]==c for layer in r["layers"].values()])) for c in config["evaluation"]}
    same=[];different=[]
    for left,right in combinations(records,2):
        if left["pair"]==right["pair"]:continue
        values=[jaccard(left["layers"][str(layer)]["oracle_top16"],right["layers"][str(layer)]["oracle_top16"]) for layer in config["layers"]]
        (same if left["family"]==right["family"] else different).extend(values)
    semantic_advantage=float(np.mean(same)-np.mean(different));corpus_family_state=[weighted_over_layers(by_corpus_family[c][f],"state_norm/32") for c in by_corpus_family for f in semantic["families"]]
    route_minus_value={f:family_fidelity[f]["routing_norm"]-family_fidelity[f]["value_norm"] for f in family_fidelity}
    hypotheses={"S1_semantic_messages_have_cooperative_source_programs":{"supported":overall_fidelity("cooperative_oracle",16)>=.80},"S2_local_state_norm_predicts_the_program":{"supported":overall_fidelity("cooperative_oracle",16)-overall_fidelity("state_norm",16)<=.10,"gap":overall_fidelity("cooperative_oracle",16)-overall_fidelity("state_norm",16)},"S3_local_state_norm_is_robust_at_32":{"supported":min(corpus_family_state)>=.75,"minimum":min(corpus_family_state)},"S4_pattern_change_alone_is_sufficient":{"supported":overall_fidelity("pattern_change",32)>=.80},"S5_value_and_routing_norms_have_distinct_regimes":{"supported":max(route_minus_value.values())>=.05 and min(route_minus_value.values())<=-.05,"differences":route_minus_value},"S6_tensor_local_rule_beats_recency":{"supported":overall_fidelity("state_norm",16)-overall_fidelity("recency",16)>=.20,"advantage":overall_fidelity("state_norm",16)-overall_fidelity("recency",16)},"S7_state_norm_transfers_to_every_semantic_family":{"supported":all(row["state_norm"]>=.80 for row in family_fidelity.values()),"family_fidelity":{f:r["state_norm"] for f,r in family_fidelity.items()}},"S8_semantic_final_messages_are_behaviorally_nontrivial":{"supported":all(value>=1e-5 for value in corpus_kl.values()),"median_kl":corpus_kl},"S9_oracle_topology_is_semantically_recurrent":{"supported":semantic_advantage>=.05,"advantage":semantic_advantage,"same":float(np.mean(same)),"different":float(np.mean(different))},"S10_semantic_program_is_not_newline_specific":{"supported":overall_fidelity("state_norm",32)>=.85,"mean_fidelity":overall_fidelity("state_norm",32)}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    public_records=[{"corpus":r["corpus"],"family":r["family"],"pair":r["pair"],"source_token":r["source_token"],"target_token":r["target_token"],"row":r["row"],"start":r["start"],"layers":r["layers"]} for r in records];OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"overall":overall,"by_corpus":by_corpus,"by_family":by_family,"by_corpus_family":by_corpus_family,"records":public_records},indent=2,allow_nan=False))
    budgets=config["budgets"];fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for method in config["methods"]:axes[0].plot(budgets,[overall_fidelity(method,b) for b in budgets],marker="o",label=method)
    axes[0].set_title("Semantic source-program frontier");x=np.arange(len(family_fidelity));width=.13
    for index,method in enumerate(("cooperative_oracle","state_norm","pattern_change","routing_norm","value_norm")):axes[1].bar(x+(index-2)*width,[family_fidelity[f][method] for f in family_fidelity],width,label=method)
    axes[1].set_xticks(x,list(family_fidelity),rotation=30,ha="right");axes[1].set_title("Rank-32 by semantic family");axes[2].bar(["same family","different family"],[np.mean(same),np.mean(different)]);axes[2].set_title("Oracle top-16 position Jaccard")
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=6) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xscale("log",base=2);axes[0].set_xlabel("source budget");axes[0].set_ylabel("Fisher fidelity");axes[1].set_ylabel("Fisher fidelity")
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Semantic transfer of the source-edge program","","Nineteen lexical substitutions across sentiment, gender, tense, scale, and polarity are evaluated at a fixed 16-token distance on FineWeb and TinyStories. The downstream cooperative oracle is compared with prediction rules using only local tensor quantities: exact source-edge state magnitude, attention-pattern change, routing magnitude, value magnitude, and fixed recency.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Semantic source transfer](bilin12_semantic_source_program_transfer.png)","","## Aggregate frontier","","| Budget | Oracle | State norm | Pattern change | Routing norm | Value norm | Recency |","|---:|---:|---:|---:|---:|---:|---:|"]
    for budget in budgets:lines.append(f"| {budget} | {overall_fidelity('cooperative_oracle',budget):.3f} | {overall_fidelity('state_norm',budget):.3f} | {overall_fidelity('pattern_change',budget):.3f} | {overall_fidelity('routing_norm',budget):.3f} | {overall_fidelity('value_norm',budget):.3f} | {overall_fidelity('recency',budget):.3f} |")
    lines += ["","## Rank-32 family transfer","","| Family | Oracle | State norm | Pattern | Routing | Value |","|---|---:|---:|---:|---:|---:|"]
    for family,row in family_fidelity.items():lines.append(f"| {family} | {row['cooperative_oracle']:.3f} | {row['state_norm']:.3f} | {row['pattern_change']:.3f} | {row['routing_norm']:.3f} | {row['value_norm']:.3f} |")
    lines += ["",f"Median full final-destination KL: FineWeb **{corpus_kl['fineweb']:.2e}**, TinyStories **{corpus_kl['tinystories']:.2e}**. Same-family versus different-family oracle-position Jaccard: **{np.mean(same):.3f}** versus **{np.mean(different):.3f}**.","","The cooperative oracle uses downstream target information. State, pattern, routing, value, and recency rankings do not. Success of a local rule establishes predictive intervention compression, not semantic meaning for any selected source position."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
