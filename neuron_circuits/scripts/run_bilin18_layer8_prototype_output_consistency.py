#!/usr/bin/env python3
"""Test whether Layer-8 prototype routes imply stable output-side verbs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_layer8_multimetric_quantization import METRICS, multimetric_site
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer8_prototype_output_consistency.json")
SOURCE=TENSOR_ROOT/"layer8_multimetric_quantization.npz"
OUTPUT=TENSOR_ROOT/"layer8_prototype_output_consistency.json"
REPORT=TENSOR_ROOT/"LAYER8_PROTOTYPE_OUTPUT_CONSISTENCY.md"
FIGURE=TENSOR_ROOT/"layer8_prototype_output_consistency.png"


def orthonormal_output_basis(effects):
    return torch.linalg.qr(effects.float(),mode="reduced").Q.T


def overlap_batch(first,second):
    products=torch.einsum("brd,bsd->brs",first,second)
    return torch.linalg.svdvals(products).square().mean(-1)


def make_pairs(labels,same,cap,rng):
    pairs=[];count=len(labels)
    if same:
        groups={value:np.flatnonzero(labels==value) for value in np.unique(labels)}
        for indices in groups.values():pairs.extend(itertools.combinations(indices.tolist(),2))
    else:
        while len(pairs)<max(cap*3,cap):
            first=int(rng.integers(count));second=int(rng.integers(count))
            if first!=second and labels[first]!=labels[second]:pairs.append((first,second))
    if len(pairs)>cap:pairs=[pairs[index] for index in rng.choice(len(pairs),cap,replace=False)]
    return np.asarray(pairs,dtype=np.int64)


def pair_summary(bases,labels,cap,seed):
    rng=np.random.default_rng(seed);result={}
    for kind,same in (("same_route",True),("different_route",False)):
        pairs=make_pairs(labels,same,cap,rng);values=[]
        for start in range(0,len(pairs),32):
            current=pairs[start:start+32];values.extend(overlap_batch(bases[current[:,0]],bases[current[:,1]]).cpu().tolist())
        result[kind]={"pairs":len(pairs),"median":float(np.median(values)),"mean":float(np.mean(values))}
    result["advantage"]=result["same_route"]["median"]-result["different_route"]["median"]
    return result


def cross_summary(first_bases,first_labels,second_bases,second_labels,cap,seed):
    rng=np.random.default_rng(seed);result={}
    for kind,same in (("same_route",True),("different_route",False)):
        pairs=[]
        while len(pairs)<cap:
            first=int(rng.integers(len(first_labels)));second=int(rng.integers(len(second_labels)))
            if (first_labels[first]==second_labels[second])==same:pairs.append((first,second))
        pairs=np.asarray(pairs);values=[]
        for start in range(0,len(pairs),32):
            current=pairs[start:start+32];values.extend(overlap_batch(first_bases[current[:,0]],second_bases[current[:,1]]).cpu().tolist())
        result[kind]={"median":float(np.median(values)),"mean":float(np.mean(values))}
    result["advantage"]=result["same_route"]["median"]-result["different_route"]["median"]
    return result


def main():
    config=json.loads(CONFIG.read_text());arrays=np.load(SOURCE);basis=torch.from_numpy(arrays["basis"]).cuda().float();discovery_axes=torch.from_numpy(arrays["axes_direct_mlp"]).cuda().float();prototypes=torch.from_numpy(arrays["prototypes_direct_mlp"]).cuda().float();model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();spec=config["prototype_discovery"];states,_=collect(model,8,spec,{**config,"positions":[config["causal_position"]]});features=states@basis;features=features/features.norm(dim=1,keepdim=True).clamp_min(1e-30);labels=grassmann_similarity(discovery_axes,prototypes).argmax(1);centroids=state_centroids(features,labels,config["clusters"]);private={};results={}
    for corpus_index,(corpus,spec) in enumerate(config["evaluation"].items()):
        source=np.load(spec["path"],mmap_mode="r");rows=source[spec["start"]:spec["start"]+spec["rows"]];route_labels=[];bases={name:[] for name in METRICS}
        for index,row in enumerate(rows):
            obj=multimetric_site(model,basis,row,config["causal_position"],config["local_rank"]);route=int((centroids@obj["feature"]).argmax());route_labels.append(route);prototype=prototypes[route]
            for name in METRICS:bases[name].append(orthonormal_output_basis(obj["responses"][name]@prototype))
            if (index+1)%16==0:print(corpus,index+1,"/",len(rows),flush=True)
        route_labels=np.asarray(route_labels);bases={name:torch.stack(value) for name,value in bases.items()};private[corpus]={"labels":route_labels,"bases":bases};results[corpus]={"route_effective_support":float(np.exp(-(np.bincount(route_labels,minlength=config["clusters"])/len(route_labels)*np.log(np.clip(np.bincount(route_labels,minlength=config["clusters"])/len(route_labels),1e-30,None))).sum())),"metrics":{name:pair_summary(value,route_labels,config["pair_cap"],config["seed"]+corpus_index*10+mi) for mi,(name,value) in enumerate(bases.items())}}
    cross_corpus={}
    for index,corpus in enumerate(("wikitext","tinystories")):
        cross_corpus[corpus]={name:cross_summary(private["fineweb"]["bases"][name],private["fineweb"]["labels"],private[corpus]["bases"][name],private[corpus]["labels"],config["pair_cap"],config["seed"]+100+index*10+mi) for mi,name in enumerate(METRICS)}
    v1=all(r["metrics"]["direct_mlp"]["same_route"]["median"]>=.5 for r in results.values());v2=all(r["metrics"]["direct_mlp"]["advantage"]>=.1 for r in results.values());v3=all(r["metrics"]["unweighted_logit"]["same_route"]["median"]>=.25 for r in results.values());v4=all(r["metrics"]["softmax_fisher"]["same_route"]["median"]>=.1 for r in results.values());v5=all(cross_corpus[c]["direct_mlp"]["advantage"]>=.1 for c in cross_corpus);v6=all(r["metrics"]["direct_mlp"]["same_route"]["median"]>=r["metrics"]["softmax_fisher"]["same_route"]["median"]+.2 for r in results.values())
    hypotheses={"V1_same_route_has_stable_direct_write_role":{"supported":v1},"V2_route_explains_direct_output_recurrence":{"supported":v2},"V3_same_route_logit_role_recurs":{"supported":v3},"V4_same_route_fisher_role_recurs":{"supported":v4},"V5_direct_role_transfers_cross_corpus":{"supported":v5},"V6_suffix_reduces_recurrence":{"supported":v6}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results,"cross_corpus":cross_corpus},indent=2,allow_nan=False));lines=["# Output-role consistency of routed Layer-8 prototypes","","The frozen state router selects one of 64 direct-input prototypes. For every selected prototype and context, its four directions are propagated through the local MLP Jacobian, unweighted suffix-logit Jacobian, and softmax-Fisher response. Output subspaces are compared within versus across route labels, including FineWeb-to-OOD pairs.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Prototype output consistency](layer8_prototype_output_consistency.png)","","| Corpus | Metric | Same/different overlap | Advantage | Pairs |","|---|---|---:|---:|---:|"]
    for corpus,result in results.items():
        for name in METRICS:
            row=result["metrics"][name];lines.append(f"| {corpus} | {name} | {row['same_route']['median']:.3f}/{row['different_route']['median']:.3f} | {row['advantage']:.3f} | {row['same_route']['pairs']} |")
    lines += ["","| Cross-corpus | Metric | Same/different overlap | Advantage |","|---|---|---:|---:|"]
    for corpus,result in cross_corpus.items():
        for name in METRICS:lines.append(f"| FineWeb→{corpus} | {name} | {result[name]['same_route']['median']:.3f}/{result[name]['different_route']['median']:.3f} | {result[name]['advantage']:.3f} |")
    lines += ["","Stable direct output roles would justify calling prototypes verbs. Stable input gain with unstable output roles instead makes them sensitivity addresses whose realized transformation remains state-dependent."]
    REPORT.write_text("\n".join(lines)+"\n");figure,plots=plt.subplots(1,3,figsize=(15,4.2));corpora=list(results);x=np.arange(len(corpora));width=.25
    for mi,name in enumerate(METRICS):plots[0].bar(x+(mi-1)*width,[results[c]["metrics"][name]["same_route"]["median"] for c in corpora],width,label=name);plots[1].bar(x+(mi-1)*width,[results[c]["metrics"][name]["advantage"] for c in corpora],width,label=name)
    for name in METRICS:plots[2].plot(list(cross_corpus),[cross_corpus[c][name]["advantage"] for c in cross_corpus],marker="o",label=name)
    plots[0].set_title("Same-route output overlap");plots[1].set_title("Within-minus-between advantage");plots[2].set_title("Cross-corpus advantage")
    for plot in plots:plot.set_xticks(x if plot is not plots[2] else np.arange(len(cross_corpus)),corpora if plot is not plots[2] else list(cross_corpus));plot.grid(alpha=.2);plot.legend(fontsize=8)
    figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
