#!/usr/bin/env python3
"""Quantize Layer-8 state-conditioned Fisher subspaces into routed prototypes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_behavioral_bottleneck import normalized_rows
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, bilinear_mlp_jvp, output_fisher_response
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_local_axis_validation import finite_directions
from run_bilin18_layer8_router_causal_validation import exact_kl, prefix_to_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG=Path("configs/bilin18_layer8_conditional_subspace_quantization.json")
OUTPUT=TENSOR_ROOT/"layer8_conditional_subspace_quantization.json"
ARRAYS=TENSOR_ROOT/"layer8_conditional_subspace_quantization.npz"
REPORT=TENSOR_ROOT/"LAYER8_CONDITIONAL_SUBSPACE_QUANTIZATION.md"
FIGURE=TENSOR_ROOT/"layer8_conditional_subspace_quantization.png"


def grassmann_similarity(first,second):
    return torch.einsum("npr,kps->nkrs",first,second).square().sum((2,3))/first.shape[2]


def grassmann_kmeans(axes,clusters,iterations):
    count=axes.shape[0]; selected=[0]
    while len(selected)<clusters:
        similarity=grassmann_similarity(axes,axes[selected]).max(1).values
        selected.append(int(similarity.argmin()))
    prototypes=axes[selected].clone(); labels=torch.zeros(count,dtype=torch.long,device=axes.device)
    for _ in range(iterations):
        similarity=grassmann_similarity(axes,prototypes); new_labels=similarity.argmax(1)
        updated=[]
        for cluster in range(clusters):
            members=axes[new_labels==cluster]
            if len(members)==0: updated.append(prototypes[cluster]); continue
            matrix=members.permute(1,0,2).reshape(axes.shape[1],-1)
            updated.append(torch.linalg.svd(matrix,full_matrices=False).U[:,:axes.shape[2]])
        updated=torch.stack(updated)
        if torch.equal(new_labels,labels): prototypes=updated; labels=new_labels; break
        prototypes=updated; labels=new_labels
    return prototypes,labels


def prototype_direction(response,prototype):
    reduced=response@prototype; gram=reduced.T@reduced; _,vectors=torch.linalg.eigh(gram); direction=prototype@vectors[:,-1]
    return direction/direction.norm().clamp_min(1e-30)


def state_centroids(features,labels,clusters):
    rows=[]
    for cluster in range(clusters):
        value=features[labels==cluster].mean(0); rows.append(value/value.norm().clamp_min(1e-30))
    return torch.stack(rows)


@torch.inference_mode()
def site_object(model,basis,row,position,rank):
    ids=torch.as_tensor(np.asarray(row[:position+1],dtype=np.int64),device="cuda")[None]; x0,value_bus,pre,normalized=prefix_to_layer(model,ids,8); block=model.blocks[8]
    native_write=block.mlp.Down(block.mlp.Left(normalized)*block.mlp.Right(normalized))+block.mlp.Down_bias; x8=pre+native_write; x=normalized[0,position].float(); tangents=bilinear_mlp_jvp(block,x,basis.T); logits,logit_tangents,_=analytic_suffix(model,x8,x0,value_bus,position,tangents,include_attention=True); response=output_fisher_response(logits,logit_tangents); values,vectors=gram_eigensystem(response); feature=x@basis; feature=feature/feature.norm().clamp_min(1e-30)
    return {"x0":x0,"value_bus":value_bus,"pre":pre,"normalized":normalized,"response":response,"values":values,"axes":vectors[:,:rank],"feature":feature}


def capture(response,prototype,optimal_energy):
    return float((response@prototype).square().sum()/optimal_energy.clamp_min(1e-30))


def summarize(records,clusters):
    result={}
    for cluster in clusters:
        rows=[r[str(cluster)] for r in records]
        result[str(cluster)]={key:float(np.median([r[key] for r in rows])) for key in ("oracle_capture","routed_capture","random_capture","finite_fidelity","finite_cosine","finite_kl_ratio")}
        result[str(cluster)]["route_match"]=float(np.mean([r["route_match"] for r in rows]))
    return result


def main():
    config=json.loads(CONFIG.read_text()); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); scan={**config,"positions":[config["causal_position"]]}
    discovery_x,_=collect(model,8,config["tensor_discovery"],{**config,"positions":[7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127]}); _,_,vh=torch.linalg.svd(discovery_x,full_matrices=False); basis=vh.T[:,:config["dimension"]]
    source=np.load(config["prototype_discovery"]["path"],mmap_mode="r"); rows=source[config["prototype_discovery"]["start"]:config["prototype_discovery"]["start"]+config["prototype_discovery"]["rows"]]; discovery_axes=[]; discovery_features=[]
    for index,row in enumerate(rows):
        obj=site_object(model,basis,row,config["causal_position"],config["local_rank"]); discovery_axes.append(obj["axes"]); discovery_features.append(obj["feature"])
        if (index+1)%8==0: print("prototype discovery",index+1,"/",len(rows),flush=True)
    discovery_axes=torch.stack(discovery_axes); discovery_features=torch.stack(discovery_features); prototypes={}; centroids={}; labels={}
    for clusters in config["clusters"]:
        prototypes[clusters],labels[clusters]=grassmann_kmeans(discovery_axes,clusters,config["kmeans_iterations"]); centroids[clusters]=state_centroids(discovery_features,labels[clusters],clusters)
    generator=torch.Generator(device="cuda").manual_seed(config["seed"]); random_prototypes={k:torch.linalg.qr(torch.randn(k,config["dimension"],config["local_rank"],generator=generator,device="cuda")).Q for k in config["clusters"]}; results={}
    for corpus_index,(corpus,spec) in enumerate(config["evaluation"].items()):
        source=np.load(spec["path"],mmap_mode="r"); eval_rows=source[spec["start"]:spec["start"]+spec["rows"]]; records=[]
        for index,row in enumerate(eval_rows):
            obj=site_object(model,basis,row,config["causal_position"],config["local_rank"]); response=obj["response"]; optimal=obj["values"][:config["local_rank"]].sum(); local_direction=obj["axes"][:,0]; record={}; directions=[local_direction]; route_meta=[]
            for clusters in config["clusters"]:
                proto=prototypes[clusters]; captures=torch.asarray([capture(response,p,optimal) for p in proto],device="cuda"); oracle=int(captures.argmax()); routed=int((centroids[clusters]@obj["feature"]).argmax()); random_capture=max(capture(response,p,optimal) for p in random_prototypes[clusters]); routed_direction=prototype_direction(response,proto[routed]); oracle_direction=prototype_direction(response,proto[oracle]); local_response=response@local_direction
                for candidate in (routed_direction,oracle_direction):
                    if float(local_response@(response@candidate))<0:candidate.mul_(-1)
                directions.extend((routed_direction,oracle_direction)); route_meta.append((clusters,oracle,routed,float(captures[oracle]),float(captures[routed]),random_capture))
            coefficient_directions=torch.stack(directions,dim=1); actual_directions=basis@coefficient_directions; finite=finite_directions(model,model.blocks[8],obj["pre"],obj["normalized"],obj["x0"],obj["value_bus"],config["causal_position"],actual_directions,[config["edit_norm_fraction"]])[config["edit_norm_fraction"]]; native_logits,changed_logits=finite; local_effect=changed_logits[0]-native_logits; local_kl=exact_kl(native_logits,changed_logits[0])
            for setting_index,(clusters,oracle,routed,oracle_capture,routed_capture,random_capture) in enumerate(route_meta):
                routed_effect=changed_logits[1+2*setting_index]-native_logits; fidelity,cosine,_=fisher_fidelity(native_logits,local_effect,routed_effect); routed_kl=exact_kl(native_logits,changed_logits[1+2*setting_index]); record[str(clusters)]={"oracle_capture":oracle_capture,"routed_capture":routed_capture,"random_capture":random_capture,"route_match":oracle==routed,"finite_fidelity":fidelity,"finite_cosine":cosine,"finite_kl_ratio":routed_kl/max(local_kl,1e-30),"oracle_cluster":oracle,"routed_cluster":routed}
            records.append(record); print(corpus,index+1,"/",len(eval_rows),flush=True)
        results[corpus]={"summary":summarize(records,config["clusters"]),"records":records}
    row16=lambda corpus:results[corpus]["summary"]["16"]; m1=all(row16(c)["oracle_capture"]>=.5 for c in results); m2=all(row16(c)["routed_capture"]>=.4 for c in results); m3=all(row16(c)["finite_fidelity"]>=.5 for c in results); m4=all(row16(c)["routed_capture"]>=2*results[c]["summary"]["1"]["routed_capture"] for c in results); m5=all(row16(c)["routed_capture"]>=row16("fineweb")["routed_capture"]-.15 for c in ("wikitext","tinystories")); m6=all(row16(c)["route_match"]>=.4 for c in results)
    hypotheses={"M1_sixteen_prototypes_cover_local_subspaces":{"supported":m1},"M2_state_router_recovers_prototype":{"supported":m2},"M3_routed_prototype_has_finite_causal_fidelity":{"supported":m3},"M4_quantization_beats_single_global_space":{"supported":m4},"M5_middle_conditional_code_transfers_ood":{"supported":m5},"M6_state_router_matches_oracle_cluster":{"supported":m6}}
    np.savez_compressed(ARRAYS,basis=basis.cpu().numpy(),**{f"prototypes_{k}":v.cpu().numpy() for k,v in prototypes.items()},**{f"centroids_{k}":v.cpu().numpy() for k,v in centroids.items()}); OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Conditional quantization of Layer-8 local Fisher subspaces","","Exact suffix Jacobians define a rank-four optimal input-sensitivity subspace at each state. FineWeb discovery subspaces are Grassmann-clustered into 1, 4, or 16 prototypes. A frozen nearest-centroid router sees only the current Layer-8 state; an oracle chooses the best prototype using the held-out Fisher Gram. Leading prototype directions are validated by exact 1%-norm interventions through layers 8--17.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Layer-8 conditional quantization](layer8_conditional_subspace_quantization.png)","","| Corpus | K | Oracle/routed/random capture | Route match | Finite fidelity/cosine | KL ratio to local |","|---|---:|---:|---:|---:|---:|"]
    for corpus,result in results.items():
        for clusters in config["clusters"]:
            row=result["summary"][str(clusters)];lines.append(f"| {corpus} | {clusters} | {row['oracle_capture']:.3f}/{row['routed_capture']:.3f}/{row['random_capture']:.3f} | {row['route_match']:.3f} | {row['finite_fidelity']:.3f}/{row['finite_cosine']:.3f} | {row['finite_kl_ratio']:.3f} |")
    lines += ["","This distinguishes three levels: per-state compactness, finite vocabulary quantizability, and predictable routing from native state. Failure at the latter two does not negate the exact local rank result."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(15,4.3)); clusters=config["clusters"]
    for corpus,result in results.items():
        plots[0].plot(clusters,[result["summary"][str(k)]["oracle_capture"] for k in clusters],marker="o",label=corpus);plots[1].plot(clusters,[result["summary"][str(k)]["routed_capture"] for k in clusters],marker="o",label=corpus);plots[2].plot(clusters,[result["summary"][str(k)]["finite_fidelity"] for k in clusters],marker="o",label=corpus)
    for plot,title in zip(plots,("Oracle prototype capture","State-routed capture","Finite causal fidelity")):plot.set_xscale("log",base=2);plot.set_xticks(clusters,clusters);plot.set_title(title);plot.grid(alpha=.2);plot.legend(fontsize=8)
    figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,ARRAYS,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
