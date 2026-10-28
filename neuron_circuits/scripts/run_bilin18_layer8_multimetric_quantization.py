#!/usr/bin/env python3
"""Locate where Layer-8 local-axis nonquantizability enters the suffix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import (
    capture, grassmann_kmeans, site_object, state_centroids,
)
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, bilinear_mlp_jvp, output_fisher_response
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_router_causal_validation import prefix_to_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer8_multimetric_quantization.json")
OUTPUT=TENSOR_ROOT/"layer8_multimetric_quantization.json"
ARRAYS=TENSOR_ROOT/"layer8_multimetric_quantization.npz"
REPORT=TENSOR_ROOT/"LAYER8_MULTIMETRIC_QUANTIZATION.md"
FIGURE=TENSOR_ROOT/"layer8_multimetric_quantization.png"
METRICS=("direct_mlp","unweighted_logit","softmax_fisher")


@torch.inference_mode()
def multimetric_site(model,basis,row,position,rank):
    ids=torch.as_tensor(np.asarray(row[:position+1],dtype=np.int64),device="cuda")[None];x0,value_bus,pre,normalized=prefix_to_layer(model,ids,8);block=model.blocks[8];native_write=block.mlp.Down(block.mlp.Left(normalized)*block.mlp.Right(normalized))+block.mlp.Down_bias;x8=pre+native_write;x=normalized[0,position].float();write_tangents=bilinear_mlp_jvp(block,x,basis.T);logits,logit_tangents,_=analytic_suffix(model,x8,x0,value_bus,position,write_tangents,include_attention=True);responses={"direct_mlp":write_tangents.T,"unweighted_logit":logit_tangents.T,"softmax_fisher":output_fisher_response(logits,logit_tangents)};result={"feature":(x@basis)/(x@basis).norm().clamp_min(1e-30),"responses":responses,"axes":{},"values":{}}
    for name,response in responses.items():values,vectors=gram_eigensystem(response);result["axes"][name]=vectors[:,:rank];result["values"][name]=values
    return result


def subspace_overlap(first,second):
    return float(torch.linalg.svdvals(first.T@second).square().mean())


def main():
    config=json.loads(CONFIG.read_text());model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();discovery_x,_=collect(model,8,config["tensor_discovery"],{**config,"positions":[7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127]});_,_,vh=torch.linalg.svd(discovery_x,full_matrices=False);basis=vh.T[:,:config["dimension"]]
    source=np.load(config["prototype_discovery"]["path"],mmap_mode="r");rows=source[config["prototype_discovery"]["start"]:config["prototype_discovery"]["start"]+config["prototype_discovery"]["rows"]];axes={name:[] for name in METRICS};features=[]
    for index,row in enumerate(rows):
        obj=multimetric_site(model,basis,row,config["causal_position"],config["local_rank"]);features.append(obj["feature"])
        for name in METRICS:axes[name].append(obj["axes"][name])
        if (index+1)%8==0:print("discovery",index+1,"/",len(rows),flush=True)
    features=torch.stack(features);axes={name:torch.stack(value) for name,value in axes.items()};prototypes={};centroids={};training_similarity={};labels={}
    from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity
    for name in METRICS:
        prototypes[name],labels[name]=grassmann_kmeans(axes[name],config["clusters"],config["kmeans_iterations"]);centroids[name]=state_centroids(features,labels[name],config["clusters"]);training_similarity[name]=float(grassmann_similarity(axes[name],prototypes[name]).max(1).values.mean())
    results={}
    for corpus,spec in config["evaluation"].items():
        source=np.load(spec["path"],mmap_mode="r");eval_rows=source[spec["start"]:spec["start"]+spec["rows"]];records=[]
        for index,row in enumerate(eval_rows):
            obj=multimetric_site(model,basis,row,config["causal_position"],config["local_rank"]);record={"metrics":{},"cross_metric_overlap":{}}
            for name in METRICS:
                response=obj["responses"][name];optimal=obj["values"][name][:config["local_rank"]].sum();values=torch.tensor([capture(response,p,optimal) for p in prototypes[name]],device="cuda");oracle=int(values.argmax());routed=int((centroids[name]@obj["feature"]).argmax());record["metrics"][name]={"oracle_capture":float(values[oracle]),"routed_capture":float(values[routed]),"route_match":oracle==routed}
            for first in METRICS:
                for second in METRICS:
                    if first<second:record["cross_metric_overlap"][f"{first}__{second}"]=subspace_overlap(obj["axes"][first],obj["axes"][second])
            records.append(record);print(corpus,index+1,"/",len(eval_rows),flush=True)
        summary={"metrics":{},"cross_metric_overlap":{}}
        for name in METRICS:summary["metrics"][name]={"oracle_capture":float(np.median([r["metrics"][name]["oracle_capture"] for r in records])),"routed_capture":float(np.median([r["metrics"][name]["routed_capture"] for r in records])),"route_match":float(np.mean([r["metrics"][name]["route_match"] for r in records]))}
        for pair in records[0]["cross_metric_overlap"]:summary["cross_metric_overlap"][pair]=float(np.median([r["cross_metric_overlap"][pair] for r in records]))
        results[corpus]={"summary":summary,"records":records}
    metric=lambda c,n:results[c]["summary"]["metrics"][n];t1=all(metric(c,"direct_mlp")["oracle_capture"]>=.25 for c in results);t2=all(metric(c,"unweighted_logit")["oracle_capture"]>=.25 for c in results);t3=all(metric(c,"softmax_fisher")["oracle_capture"]<.1 for c in results);t4=all(metric(c,"direct_mlp")["oracle_capture"]>=metric(c,"softmax_fisher")["oracle_capture"]+.1 for c in results);t5=all(metric(c,"direct_mlp")["routed_capture"]>=.15 for c in results);t6=all(max(metric(c,n)["oracle_capture"] for c in results)-min(metric(c,n)["oracle_capture"] for c in results)<=.1 for n in METRICS)
    hypotheses={"T1_direct_tensor_axes_are_quantizable":{"supported":t1},"T2_unweighted_logit_axes_are_quantizable":{"supported":t2},"T3_fisher_axes_remain_nonquantizable":{"supported":t3},"T4_downstream_geometry_creates_variability":{"supported":t4},"T5_native_state_routes_direct_axes":{"supported":t5},"T6_metric_regime_transfers":{"supported":t6}}
    np.savez_compressed(ARRAYS,basis=basis.cpu().numpy(),**{f"axes_{n}":v.cpu().numpy() for n,v in axes.items()},**{f"prototypes_{n}":v.cpu().numpy() for n,v in prototypes.items()});OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"training_similarity":training_similarity,"results":results},indent=2,allow_nan=False))
    lines=["# Multimetric localization of Layer-8 nonquantizability","","The same native Layer-8 state basis and 64-prototype budget are applied separately to the direct MLP write Jacobian, the full unweighted suffix-logit Jacobian, and the softmax-Fisher-weighted suffix Jacobian. This locates whether local-axis variability originates in the tensor layer, downstream transport/readout, or probability geometry.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Multimetric quantization](layer8_multimetric_quantization.png)","","| Corpus | Metric | Training similarity | Oracle capture | Routed capture | Route match |","|---|---|---:|---:|---:|---:|"]
    for corpus,result in results.items():
        for name in METRICS:
            row=result["summary"]["metrics"][name];lines.append(f"| {corpus} | {name} | {training_similarity[name]:.3f} | {row['oracle_capture']:.3f} | {row['routed_capture']:.3f} | {row['route_match']:.3f} |")
    lines += ["","Cross-metric same-site overlaps are included in JSON. If direct axes quantize but Fisher axes do not, downstream geometry creates the continuous regime. If all fail, the native bilinear Jacobian itself varies continuously with state."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(15,4.2));x=np.arange(len(METRICS));width=.25
    for ci,(corpus,result) in enumerate(results.items()):plots[0].bar(x+(ci-1)*width,[result["summary"]["metrics"][n]["oracle_capture"] for n in METRICS],width,label=corpus);plots[1].bar(x+(ci-1)*width,[result["summary"]["metrics"][n]["routed_capture"] for n in METRICS],width,label=corpus)
    plots[2].bar(METRICS,[training_similarity[n] for n in METRICS]);plots[0].set_title("Held-out oracle capture");plots[1].set_title("State-routed capture");plots[2].set_title("Discovery clustering fit")
    for plot in plots:plot.set_xticks(x,METRICS,rotation=20);plot.grid(alpha=.2)
    plots[0].legend();plots[1].legend();figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,ARRAYS,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
