#!/usr/bin/env python3
"""High-resolution red team of Layer-8 local-subspace quantization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import (
    capture, grassmann_kmeans, grassmann_similarity, site_object, state_centroids,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer8_conditional_quantization_redteam.json")
OUTPUT=TENSOR_ROOT/"layer8_conditional_quantization_redteam.json"
ARRAYS=TENSOR_ROOT/"layer8_conditional_quantization_redteam.npz"
REPORT=TENSOR_ROOT/"LAYER8_CONDITIONAL_QUANTIZATION_REDTEAM.md"
FIGURE=TENSOR_ROOT/"layer8_conditional_quantization_redteam.png"


def farthest_medoids(axes,count):
    selected=[0]
    while len(selected)<count:
        similarity=grassmann_similarity(axes,axes[selected]).max(1).values
        selected.append(int(similarity.argmin()))
    return axes[selected],selected


def bank_metrics(response,optimal,feature,prototypes,centroids):
    values=torch.tensor([capture(response,p,optimal) for p in prototypes],device="cuda")
    oracle=int(values.argmax()); routed=int((centroids@feature).argmax())
    return {"oracle_capture":float(values[oracle]),"routed_capture":float(values[routed]),"oracle":oracle,"routed":routed,"route_match":oracle==routed}


def summarize(records,clusters):
    result={}
    for count in clusters:
        rows=[row[str(count)] for row in records]; result[str(count)]={}
        for kind in ("centroid","medoid"):
            result[str(count)][kind]={"oracle_capture":float(np.median([r[kind]["oracle_capture"] for r in rows])),"routed_capture":float(np.median([r[kind]["routed_capture"] for r in rows])),"route_match":float(np.mean([r[kind]["route_match"] for r in rows]))}
        result[str(count)]["random_oracle_capture"]=float(np.median([r["random_oracle_capture"] for r in rows]))
    result["nearest_context_capture"]=float(np.median([row["nearest_context_capture"] for row in records]))
    return result


def main():
    config=json.loads(CONFIG.read_text()); model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval(); discovery_x,_=collect(model,8,config["tensor_discovery"],{**config,"positions":[7,15,23,31,39,47,55,63,71,79,87,95,103,111,119,127]});_,_,vh=torch.linalg.svd(discovery_x,full_matrices=False);basis=vh.T[:,:config["dimension"]]
    source=np.load(config["prototype_discovery"]["path"],mmap_mode="r");rows=source[config["prototype_discovery"]["start"]:config["prototype_discovery"]["start"]+config["prototype_discovery"]["rows"]];axes=[];features=[]
    for index,row in enumerate(rows):
        obj=site_object(model,basis,row,config["causal_position"],config["local_rank"]);axes.append(obj["axes"]);features.append(obj["feature"])
        if (index+1)%16==0:print("discovery",index+1,"/",len(rows),flush=True)
    axes=torch.stack(axes);features=torch.stack(features);centroid_banks={};medoid_banks={};centroid_routes={};medoid_routes={};training_similarity={}
    for count in config["clusters"]:
        centroid,labels=grassmann_kmeans(axes,count,config["kmeans_iterations"]);medoid,_=farthest_medoids(axes,count);medoid_labels=grassmann_similarity(axes,medoid).argmax(1);centroid_banks[count]=centroid;medoid_banks[count]=medoid;centroid_routes[count]=state_centroids(features,labels,count);medoid_routes[count]=state_centroids(features,medoid_labels,count);training_similarity[str(count)]={"centroid":float(grassmann_similarity(axes,centroid).max(1).values.mean()),"medoid":float(grassmann_similarity(axes,medoid).max(1).values.mean())}
    generator=torch.Generator(device="cuda").manual_seed(config["seed"]);random_banks={count:torch.linalg.qr(torch.randn(count,config["dimension"],config["local_rank"],generator=generator,device="cuda")).Q for count in config["clusters"]};results={}
    for corpus,spec in config["evaluation"].items():
        source=np.load(spec["path"],mmap_mode="r");eval_rows=source[spec["start"]:spec["start"]+spec["rows"]];records=[]
        for index,row in enumerate(eval_rows):
            obj=site_object(model,basis,row,config["causal_position"],config["local_rank"]);response=obj["response"];optimal=obj["values"][:config["local_rank"]].sum();nearest=int((features@obj["feature"]).argmax());record={"nearest_context_capture":capture(response,axes[nearest],optimal)}
            for count in config["clusters"]:
                centroid=bank_metrics(response,optimal,obj["feature"],centroid_banks[count],centroid_routes[count]);medoid=bank_metrics(response,optimal,obj["feature"],medoid_banks[count],medoid_routes[count]);random_capture=max(capture(response,p,optimal) for p in random_banks[count]);record[str(count)]={"centroid":centroid,"medoid":medoid,"random_oracle_capture":random_capture}
            records.append(record);print(corpus,index+1,"/",len(eval_rows),flush=True)
        results[corpus]={"summary":summarize(records,config["clusters"]),"records":records}
    row=lambda corpus:results[corpus]["summary"]["64"];r1=all(row(c)["medoid"]["oracle_capture"]>=.25 for c in results);r2=all(max(row(c)["centroid"]["oracle_capture"],row(c)["medoid"]["oracle_capture"])>=2*row(c)["random_oracle_capture"] for c in results);r3=all(max(row(c)["centroid"]["routed_capture"],row(c)["medoid"]["routed_capture"])>=.15 for c in results);r4=all(results[c]["summary"]["nearest_context_capture"]>=.15 for c in results);r5=all(row(c)["medoid"]["oracle_capture"]>=2*results[c]["summary"]["1"]["medoid"]["oracle_capture"] for c in results);r6=all(abs(row(c)["medoid"]["oracle_capture"]-row("fineweb")["medoid"]["oracle_capture"])<=.1 for c in ("wikitext","tinystories"))
    hypotheses={"R1_sixtyfour_medoids_cover_local_axes":{"supported":r1},"R2_learned_prototypes_beat_random":{"supported":r2},"R3_state_router_recovers_useful_capture":{"supported":r3},"R4_nearest_context_predicts_local_subspace":{"supported":r4},"R5_more_prototypes_improve_coverage":{"supported":r5},"R6_negative_or_positive_regime_transfers":{"supported":r6}}
    np.savez_compressed(ARRAYS,basis=basis.cpu().numpy(),discovery_axes=axes.cpu().numpy(),discovery_features=features.cpu().numpy(),**{f"centroid_{k}":v.cpu().numpy() for k,v in centroid_banks.items()},**{f"medoid_{k}":v.cpu().numpy() for k,v in medoid_banks.items()});OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"training_similarity":training_similarity,"results":results},indent=2,allow_nan=False))
    lines=["# High-resolution red team of Layer-8 conditional quantization","","The discovery set is expanded to 256 exact local Fisher subspaces and the bank to 64 prototypes. Grassmann centroids are compared with actual discovery medoids, matched random subspaces, state-centroid routing, oracle prototype selection, and a nearest-native-context router.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![High-resolution quantization](layer8_conditional_quantization_redteam.png)","","| Corpus | K | Centroid oracle/routed | Medoid oracle/routed | Random oracle | Route match C/M |","|---|---:|---:|---:|---:|---:|"]
    for corpus,result in results.items():
        for count in config["clusters"]:
            row=result["summary"][str(count)];lines.append(f"| {corpus} | {count} | {row['centroid']['oracle_capture']:.3f}/{row['centroid']['routed_capture']:.3f} | {row['medoid']['oracle_capture']:.3f}/{row['medoid']['routed_capture']:.3f} | {row['random_oracle_capture']:.3f} | {row['centroid']['route_match']:.3f}/{row['medoid']['route_match']:.3f} |")
        lines.append(f"| {corpus} | nearest context | — | — | — | capture {result['summary']['nearest_context_capture']:.3f} |")
    lines += ["","Training-set Grassmann similarity verifies that clustering itself fits its declared objective. Held-out failure would therefore identify nonrecurrent downstream-conditioned axes rather than an implementation failure."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(15,4.3));counts=config["clusters"]
    for corpus,result in results.items():
        plots[0].plot(counts,[result["summary"][str(k)]["medoid"]["oracle_capture"] for k in counts],marker="o",label=corpus);plots[1].plot(counts,[result["summary"][str(k)]["centroid"]["routed_capture"] for k in counts],marker="o",label=corpus);plots[2].plot(counts,[result["summary"][str(k)]["random_oracle_capture"] for k in counts],marker="o",label=corpus)
    for plot,title in zip(plots,("Medoid oracle coverage","State-routed centroid","Random-bank oracle")):plot.set_xscale("log",base=2);plot.set_xticks(counts,counts);plot.set_title(title);plot.grid(alpha=.2);plot.legend(fontsize=8)
    figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,ARRAYS,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
