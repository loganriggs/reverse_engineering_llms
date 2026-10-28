#!/usr/bin/env python3
"""Finite causal red team of routed direct-MLP Layer-8 prototypes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import capture, grassmann_similarity, state_centroids
from run_bilin18_layer8_direct_tensor_jacobian import bilinear_mlp_jvp
from run_bilin18_layer8_local_axis_validation import finite_directions
from run_bilin18_layer8_router_causal_validation import prefix_to_layer
from run_bilin18_layer8_multimetric_quantization import multimetric_site
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer8_direct_prototype_causal_redteam.json")
SOURCE=TENSOR_ROOT/"layer8_multimetric_quantization.npz"
OUTPUT=TENSOR_ROOT/"layer8_direct_prototype_causal_redteam.json"
REPORT=TENSOR_ROOT/"LAYER8_DIRECT_PROTOTYPE_CAUSAL_REDTEAM.md"
FIGURE=TENSOR_ROOT/"layer8_direct_prototype_causal_redteam.png"


def row_subspace_overlap(first,second):
    first_v=torch.linalg.svd(first.float(),full_matrices=False).Vh[:first.shape[0]];second_v=torch.linalg.svd(second.float(),full_matrices=False).Vh[:second.shape[0]]
    return float(torch.linalg.svdvals(first_v@second_v.T).square().mean())


@torch.inference_mode()
def finite_write_effects(block,normalized,position,directions,scale):
    x=normalized[0,position].float();native=block.mlp.Down(block.mlp.Left(normalized)*block.mlp.Right(normalized))+block.mlp.Down_bias;variants=normalized.repeat(directions.shape[1],1,1);variants[:,position]=(x[None]+scale*x.norm()*directions.T).to(variants.dtype);writes=block.mlp.Down(block.mlp.Left(variants)*block.mlp.Right(variants))+block.mlp.Down_bias
    return writes[:,position].float()-native[0,position].float()


def summarize(records):
    keys=("oracle_capture","routed_capture","random_oracle_capture","finite_mlp_overlap","finite_suffix_overlap","finite_energy_ratio")
    return {key:float(np.median([row[key] for row in records])) for key in keys}


def main():
    config=json.loads(CONFIG.read_text());arrays=np.load(SOURCE);basis=torch.from_numpy(arrays["basis"]).cuda().float();discovery_axes=torch.from_numpy(arrays["axes_direct_mlp"]).cuda().float();prototypes=torch.from_numpy(arrays["prototypes_direct_mlp"]).cuda().float();model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();block=model.blocks[8]
    spec=config["prototype_discovery"];states,_=collect(model,8,spec,{**config,"positions":[config["causal_position"]]});features=states@basis;features=features/features.norm(dim=1,keepdim=True).clamp_min(1e-30);labels=grassmann_similarity(discovery_axes,prototypes).argmax(1);centroids=state_centroids(features,labels,config["clusters"]);generator=torch.Generator(device="cuda").manual_seed(config["seed"]);random_bank=torch.linalg.qr(torch.randn(config["clusters"],config["dimension"],config["local_rank"],generator=generator,device="cuda")).Q;results={}
    for corpus,spec in config["evaluation"].items():
        source=np.load(spec["path"],mmap_mode="r");rows=source[spec["start"]:spec["start"]+spec["rows"]];records=[]
        for index,row in enumerate(rows):
            obj=multimetric_site(model,basis,row,config["causal_position"],config["local_rank"]);response=obj["responses"]["direct_mlp"];optimal=obj["values"]["direct_mlp"][:config["local_rank"]].sum();captures=torch.tensor([capture(response,p,optimal) for p in prototypes],device="cuda");oracle=int(captures.argmax());routed=int((centroids@obj["feature"]).argmax());random_capture=max(capture(response,p,optimal) for p in random_bank);local_coeff=obj["axes"]["direct_mlp"];routed_coeff=prototypes[routed];coefficient_directions=torch.cat((local_coeff,routed_coeff),dim=1);directions=basis@coefficient_directions
            ids=torch.as_tensor(np.asarray(row[:config["causal_position"]+1],dtype=np.int64),device="cuda")[None];x0,value_bus,pre,normalized=prefix_to_layer(model,ids,8);write_effects=finite_write_effects(block,normalized,config["causal_position"],directions,config["edit_norm_fraction"]);local_write=write_effects[:config["local_rank"]];routed_write=write_effects[config["local_rank"]:];finite=finite_directions(model,block,pre,normalized,x0,value_bus,config["causal_position"],directions,[config["edit_norm_fraction"]])[config["edit_norm_fraction"]];native_logits,changed_logits=finite;logit_effects=changed_logits-native_logits;local_logits=logit_effects[:config["local_rank"]];routed_logits=logit_effects[config["local_rank"]:]
            records.append({"oracle_capture":float(captures[oracle]),"routed_capture":float(captures[routed]),"random_oracle_capture":random_capture,"finite_mlp_overlap":row_subspace_overlap(local_write,routed_write),"finite_suffix_overlap":row_subspace_overlap(local_logits,routed_logits),"finite_energy_ratio":float(routed_write.square().sum()/local_write.square().sum().clamp_min(1e-30)),"oracle":oracle,"routed":routed});print(corpus,index+1,"/",len(rows),flush=True)
        results[corpus]={"summary":summarize(records),"records":records}
    p1=all(r["summary"]["oracle_capture"]>=2*r["summary"]["random_oracle_capture"] for r in results.values());p2=all(r["summary"]["routed_capture"]>=.3 for r in results.values());p3=all(r["summary"]["finite_mlp_overlap"]>=.7 for r in results.values());p4=all(r["summary"]["finite_suffix_overlap"]>=.4 for r in results.values());p5=all(abs(r["summary"]["finite_energy_ratio"]-r["summary"]["routed_capture"])<=.1 for r in results.values());fine=results["fineweb"]["summary"]["finite_mlp_overlap"];p6=all(r["summary"]["finite_mlp_overlap"]>=fine-.15 for c,r in results.items() if c!="fineweb")
    hypotheses={"P1_learned_bank_beats_random":{"supported":p1},"P2_state_router_retains_direct_energy":{"supported":p2},"P3_finite_mlp_subspace_transfers":{"supported":p3},"P4_suffix_effect_partially_transfers":{"supported":p4},"P5_finite_direct_energy_calibrates":{"supported":p5},"P6_direct_prototype_causality_is_ood_stable":{"supported":p6}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False));lines=["# Finite causal red team of Layer-8 direct prototypes","","The frozen 64-prototype bank is routed only from the native Layer-8 state. Matched random subspaces test spectral specificity. At each held-out site, the local-optimal and routed rank-four coefficient spaces are edited at 1% input norm; resulting MLP-write and final-logit effect subspaces are compared after full nonlinear recomputation.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Direct prototype causal red team](layer8_direct_prototype_causal_redteam.png)","","| Corpus | Oracle/routed/random capture | Finite MLP overlap | Suffix overlap | Finite energy ratio |","|---|---:|---:|---:|---:|"]
    for corpus,result in results.items():
        row=result["summary"];lines.append(f"| {corpus} | {row['oracle_capture']:.3f}/{row['routed_capture']:.3f}/{row['random_oracle_capture']:.3f} | {row['finite_mlp_overlap']:.3f} | {row['finite_suffix_overlap']:.3f} | {row['finite_energy_ratio']:.3f} |")
    lines += ["","High native-write overlap with weak suffix overlap would validate the prototypes as layer-local tensor verbs while rejecting them as complete downstream behavioral circuits."]
    REPORT.write_text("\n".join(lines)+"\n");figure,plots=plt.subplots(1,3,figsize=(14,4.2));corpora=list(results);plots[0].bar(corpora,[results[c]["summary"]["routed_capture"] for c in corpora]);plots[0].set_title("Routed tangent capture");plots[1].bar(corpora,[results[c]["summary"]["finite_mlp_overlap"] for c in corpora]);plots[1].set_title("Finite MLP subspace overlap");plots[2].bar(corpora,[results[c]["summary"]["finite_suffix_overlap"] for c in corpora]);plots[2].set_title("Final-logit subspace overlap")
    for plot in plots:plot.grid(alpha=.2)
    figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
