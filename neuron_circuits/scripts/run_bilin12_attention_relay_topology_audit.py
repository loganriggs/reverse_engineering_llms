#!/usr/bin/env python3
"""Partition recurrent attention messages by source and destination topology."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_attention_relay_topology_audit.json")
OUTPUT=TENSOR_ROOT/"bilin12_attention_relay_topology_audit.json"
REPORT=TENSOR_ROOT/"BILIN12_ATTENTION_RELAY_TOPOLOGY_AUDIT.md"
FIGURE=TENSOR_ROOT/"bilin12_attention_relay_topology_audit.png"


def vector_shapley(names,worlds):
    count=len(names);result={}
    for index,name in enumerate(names):
        bit=1<<index;effect=torch.zeros_like(worlds[0])
        for mask in range(1<<count):
            if mask&bit:continue
            size=mask.bit_count();weight=math.factorial(size)*math.factorial(count-size-1)/math.factorial(count);effect+=weight*(worlds[mask|bit]-worlds[mask])
        result[name]=effect
    return result


def attention_mixed_value(block,details):
    return (1-block.attn.lamb)*details["current_value"]+block.attn.lamb*details["value_bus"]


@torch.inference_mode()
def source_group_write(block,details,start,end):
    pattern=details["pattern"][:,:,:,start:end];value=attention_mixed_value(block,details)[:,start:end]
    attended=torch.einsum("bhqk,bkhd->bhqd",pattern,value);batch,heads,queries,head_dimension=attended.shape
    flattened=attended.transpose(1,2).contiguous().view(batch,queries,heads*head_dimension)
    return block.attn.c_proj(flattened)


def group_worlds(base_write,deltas,names):
    worlds=[]
    for mask in range(1<<len(names)):
        value=base_write
        for index,name in enumerate(names):
            if mask&(1<<index):value=value+deltas[name]
        worlds.append(value)
    return torch.stack(worlds)


def empty_accumulator(destination_names,source_names):
    return {"examples":0,"destination":{"den":0.,"cross":{n:0. for n in destination_names},"state_den":0.,"state_cross":{n:0. for n in destination_names}},"source":{"den":0.,"cross":{n:0. for n in source_names},"state_den":0.,"state_cross":{n:0. for n in source_names}},"destination_error":0.,"source_error":0.}


def update_partition(row,names,world_writes,logits):
    write_target=world_writes[-1]-world_writes[0];write_den=write_target.square().sum().clamp_min(1e-30);row["state_den"]+=float(write_den)
    for name,effect in vector_shapley(names,world_writes).items():row["state_cross"][name]+=float((effect*write_target).sum())
    target=logits[-1]-logits[0];energy,_,_,_=fisher_terms(logits[0],target,target);row["den"]+=float(energy.sum())
    for name,effect in vector_shapley(names,logits).items():
        _,_,cross,_=fisher_terms(logits[0],target,effect);row["cross"][name]+=float(cross.sum())


def finalize_partition(row):
    return {"causal_shapley":{n:row["cross"][n]/max(row["den"],1e-30) for n in row["cross"]},"state_projection":{n:row["state_cross"][n]/max(row["state_den"],1e-30) for n in row["state_cross"]}}


@torch.inference_mode()
def evaluate(model,tokenizer,spec,config):
    rows=load_rows(spec);newline=tokenizer.encode("\n",add_special_tokens=False)[0];destination_names=list(config["destination_groups"]);source_names=list(config["source_groups"]);acc={layer:empty_accumulator(destination_names,source_names) for layer in config["layers"]}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:config["sequence_length"]],dtype=np.int64);base_ids=torch.as_tensor(raw,device="cuda");keep=base_ids[:,config["edit_position"]]!=newline
        if not bool(keep.any()):continue
        base_ids=base_ids[keep];changed_ids=base_ids.clone();changed_ids[:,config["edit_position"]]=newline;base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids)
        for layer in config["layers"]:
            block=model.blocks[layer];base=base_rows[layer];changed=changed_rows[layer];full_delta=changed["details"]["write"]-base["details"]["write"]
            destination_deltas={}
            for name,(lo,hi) in config["destination_groups"].items():
                value=torch.zeros_like(full_delta);value[:,lo:hi]=full_delta[:,lo:hi];destination_deltas[name]=value
            source_deltas={name:source_group_write(block,changed["details"],lo,hi)-source_group_write(block,base["details"],lo,hi) for name,(lo,hi) in config["source_groups"].items()}
            destination_worlds=group_worlds(base["details"]["write"],destination_deltas,destination_names);source_worlds=group_worlds(base["details"]["write"],source_deltas,source_names)
            destination_logits=logits_from_attention_writes(model,layer,base,destination_worlds);source_logits=logits_from_attention_writes(model,layer,base,source_worlds)
            update_partition(acc[layer]["destination"],destination_names,destination_worlds,destination_logits);update_partition(acc[layer]["source"],source_names,source_worlds,source_logits);acc[layer]["examples"]+=len(base_ids);acc[layer]["destination_error"]=max(acc[layer]["destination_error"],float((sum(destination_deltas.values())-full_delta).abs().max()));acc[layer]["source_error"]=max(acc[layer]["source_error"],float((sum(source_deltas.values())-full_delta).abs().max()))
    return {str(layer):{"examples":acc[layer]["examples"],"destination":finalize_partition(acc[layer]["destination"]),"source":finalize_partition(acc[layer]["source"]),"maximum_destination_recomposition_error":acc[layer]["destination_error"],"maximum_source_recomposition_error":acc[layer]["source_error"]} for layer in config["layers"]}


def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");results={}
    for corpus,spec in config["evaluation"].items():print("evaluating",corpus,flush=True);results[corpus]=evaluate(model,tokenizer,spec,config)
    destination_names=list(config["destination_groups"]);source_names=list(config["source_groups"]);rows=[results[c][str(layer)] for c in results for layer in config["layers"]]
    abs_share=lambda row,partition,name:abs(row[partition]["causal_shapley"][name])/max(sum(abs(v) for v in row[partition]["causal_shapley"].values()),1e-30)
    hypotheses={
      "T1_edge_partitions_recompose_attention":{"supported":all(max(r["maximum_destination_recomposition_error"],r["maximum_source_recomposition_error"])<=1e-4 for r in rows)},
      "T2_direct_final_write_is_not_the_main_route":{"supported":all(abs_share(r,"destination","final_32")<=.20 for r in rows)},
      "T3_early_backbone_writes_locally":{"supported":all(abs_share(results[c][str(layer)],"destination","local_0_4")+abs_share(results[c][str(layer)],"destination","relay_5_16")>=.50 for c in results for layer in (5,6))},
      "T4_relay_front_moves_forward_with_depth":{"supported":all(abs_share(results[c]["10"],"destination","late_17_31")+abs_share(results[c]["10"],"destination","final_32")>abs_share(results[c]["5"],"destination","late_17_31")+abs_share(results[c]["5"],"destination","final_32") for c in results)},
      "T5_edited_source_neighborhood_seeds_the_backbone":{"supported":all(abs_share(results[c][str(layer)],"source","edit_neighborhood")>=.50 for c in results for layer in (5,6))},
      "T6_later_sources_grow_after_relay":{"supported":all(abs_share(results[c]["10"],"source","later")>abs_share(results[c]["5"],"source","later") for c in results)},
      "T7_state_energy_and_behavioral_routing_differ":{"supported":any(abs(r[p]["causal_shapley"][n]-r[p]["state_projection"][n])>=.15 for r in rows for p,names in (("destination",destination_names),("source",source_names)) for n in names)},
      "T8_dominant_topology_transfers_ood":{"supported":all(len({max((results[c][str(layer)][partition]["causal_shapley"]),key=lambda n:abs(results[c][str(layer)][partition]["causal_shapley"][n])) for c in results})==1 for layer in config["layers"] for partition in ("destination","source"))},
    }
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    layers=np.array(config["layers"]);fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for name in destination_names:
        axes[0].plot(layers,[np.mean([results[c][str(layer)]["destination"]["causal_shapley"][name] for c in results]) for layer in layers],marker="o",label=name)
    for name in source_names:
        axes[1].plot(layers,[np.mean([results[c][str(layer)]["source"]["causal_shapley"][name] for c in results]) for layer in layers],marker="o",label=name)
    for corpus in results:
        axes[2].plot(layers,[sum(abs_share(results[corpus][str(layer)],"destination",name)*((config["destination_groups"][name][0]+config["destination_groups"][name][1]-1)/2-config["edit_position"]) for name in destination_names) for layer in layers],marker="o",label=corpus)
    axes[0].set_title("Destination causal Shapley");axes[1].set_title("Source causal Shapley");axes[2].set_title("Absolute-share destination offset")
    for axis in axes:axis.set_xlabel("layer");axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Attention relay-topology audit","","For the position-95 newline edit, each recurrent-backbone attention-write delta is partitioned two ways. Destination groups ask where the message is written: positions 95--99, 100--111, 112--126, or the final position 127. Source groups ask where its attended value originated: before 93, the edited neighborhood 93--97, or later positions 98--127. Every subset is propagated through the exact MLP and suffix.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Relay topology](bilin12_attention_relay_topology_audit.png)","","| Corpus | Layer | Destination local / relay / late / final | Source earlier / edit-near / later |","|---|---:|---:|---:|"]
    for corpus in results:
        for layer in config["layers"]:
            row=results[corpus][str(layer)];dest=" / ".join(f"{row['destination']['causal_shapley'][n]:.3f}" for n in destination_names);source=" / ".join(f"{row['source']['causal_shapley'][n]:.3f}" for n in source_names);lines.append(f"| {corpus} | {layer} | {dest} | {source} |")
    lines += ["","Destination and source partitions are message-topology explanations, not semantic labels. A local destination write can be behaviorally important because later layers relay it; a later source can already contain information copied from the original edit. Exact edge provenance and original information provenance are therefore different notions."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
