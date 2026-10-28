#!/usr/bin/env python3
"""Large-corpus semantic and weight atlas for routed Layer-17 experts."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer17_routed_semantic_atlas.json")
SOURCE_ARRAYS=TENSOR_ROOT/"layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS=TENSOR_ROOT/"layer17_tail_continuous_rank_frontier.npz"
SEED_CHECKPOINT=TENSOR_ROOT/"layer17_tail_routed_seed_stability.pt"
OUTPUT=TENSOR_ROOT/"layer17_routed_semantic_atlas.json"
REPORT=TENSOR_ROOT/"LAYER17_ROUTED_SEMANTIC_ATLAS.md"
FIGURE=TENSOR_ROOT/"layer17_routed_semantic_atlas.png"


def lookup_fit(feature,labels):
    mapping={}; default=int(np.bincount(labels).argmax())
    for value in np.unique(feature):
        subset=labels[feature==value]; mapping[int(value)]=int(np.bincount(subset).argmax())
    return mapping,default


def lookup_score(feature,labels,mapping,default):
    prediction=np.asarray([mapping.get(int(value),default) for value in feature]); seen=np.asarray([int(value) in mapping for value in feature])
    return {"accuracy":float(np.mean(prediction==labels)),"coverage":float(np.mean(seen)),"baseline":float(np.bincount(labels).max()/len(labels))}


def adjusted_rand(first,second):
    first=np.asarray(first); second=np.asarray(second); n=len(first)
    table=np.zeros((int(first.max())+1,int(second.max())+1),dtype=np.int64)
    np.add.at(table,(first,second),1)
    choose2=lambda x:x*(x-1)/2
    index=float(choose2(table).sum()); row=float(choose2(table.sum(1)).sum()); col=float(choose2(table.sum(0)).sum()); total=choose2(n)
    expected=row*col/max(total,1); maximum=.5*(row+col)
    return float((index-expected)/max(maximum-expected,1e-30))


def shape_id(text):
    stripped=text.strip()
    if "\n" in text:return 0
    if not stripped:return 1
    if stripped.isdigit():return 2
    if stripped.isalpha() and text[:1].isspace():return 3
    if stripped.isalpha():return 4
    if all(not c.isalnum() for c in stripped):return 5
    if any(ord(c)>127 for c in text):return 6
    return 7


def structural_bitmask(flags):
    weights=(1<<np.arange(flags.shape[1],dtype=np.int64))[None]
    return (flags.astype(np.int64)*weights).sum(1)


def semantic_profile(labels,current_shape,next_shape,current_flags,next_flags,position,experts):
    profiles=[]
    for expert in range(experts):
        mask=labels==expert
        pieces=[]
        for values,size in ((current_shape,8),(next_shape,8),(position,16)):
            pieces.append(np.bincount(values[mask],minlength=size)/max(mask.sum(),1))
        pieces.extend((current_flags[mask].mean(0) if mask.any() else np.zeros(current_flags.shape[1]),next_flags[mask].mean(0) if mask.any() else np.zeros(next_flags.shape[1])))
        vector=np.concatenate(pieces); profiles.append(vector/max(np.linalg.norm(vector),1e-30))
    return np.asarray(profiles)


def align_profiles(reference,candidate):
    similarity=reference@candidate.T; best=None
    for permutation in itertools.permutations(range(len(reference))):
        score=float(np.mean([similarity[i,permutation[i]] for i in range(len(reference))]))
        if best is None or score>best[0]:best=(score,permutation)
    return {"mean_cosine":best[0],"reference_to_candidate":list(best[1]),"matrix":similarity.tolist()}


def top_token_lift(token_ids,labels,expert,tokenizer,minimum=20,count=6):
    global_counts=np.bincount(token_ids,minlength=tokenizer.vocab_size); expert_counts=np.bincount(token_ids[labels==expert],minlength=tokenizer.vocab_size); eligible=np.flatnonzero(global_counts>=minimum)
    lift=(expert_counts[eligible]/max((labels==expert).sum(),1))/(global_counts[eligible]/len(labels)); order=eligible[np.argsort(lift)[::-1][:count]]
    return [{"token_id":int(token),"text":tokenizer.decode([int(token)]),"lift":float((expert_counts[token]/max((labels==expert).sum(),1))/(global_counts[token]/len(labels))),"count":int(global_counts[token])} for token in order]


def top_contexts(rows,positions,labels,confidence,expert,tokenizer,count=4):
    candidates=np.flatnonzero(labels==expert); selected=candidates[np.argsort(confidence[candidates])[::-1][:count]]; result=[]
    for index in selected:
        row=index//len(positions); position=positions[index%len(positions)]; start=max(0,position-12); stop=min(rows.shape[1],position+8)
        result.append({"confidence":float(confidence[index]),"position":int(position),"current":tokenizer.decode([int(rows[row,position])]),"context":tokenizer.decode(rows[row,start:stop].tolist()).replace("\n","\\n")})
    return result


def output_role(model,mean_score,output_basis,tokenizer,count=6):
    direction=torch.as_tensor(mean_score,device="cuda",dtype=torch.float32)@output_basis.T; logits=model.lm_head(direction[None]).float().squeeze(0); high=torch.topk(logits,count).indices.tolist(); low=torch.topk(-logits,count).indices.tolist()
    return {"promoted":[tokenizer.decode([i]) for i in high],"suppressed":[tokenizer.decode([i]) for i in low]}


def main():
    config=json.loads(CONFIG.read_text()); source=np.load(SOURCE_ARRAYS); frontier=np.load(FRONTIER_ARRAYS); checkpoint=torch.load(SEED_CHECKPOINT,map_location="cuda",weights_only=False); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    token_text=[tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]; all_flags=np.stack([token_flags(text) for text in token_text]); all_shape=np.asarray([shape_id(text) for text in token_text],dtype=np.int64)
    output_basis=torch.from_numpy(source["tail_output_basis"][:,:4]).cuda().float(); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); native=model.blocks[config["layer"]].mlp; down,left,right=native.Down.weight.float().detach(),native.Left.weight.float().detach(),native.Right.weight.float().detach()
    discovery_x,_=collect(model,config["layer"],config["discovery"],config); square,inverse,covariance=covariance_transforms(discovery_x,config["covariance_floor_fraction"]); cleaned=[]
    for component in range(4):
        coefficients=down.T@output_basis[:,component]; raw=left.T@(coefficients[:,None]*right); raw=.5*(raw+raw.T); clean,_=trace_clean(square@raw@square); cleaned.append(clean)
    _,common_basis=common_input(cleaned); candidate_basis=common_basis[:,:config["candidate_pool_rank"]]; forms=torch.einsum("di,kde,ej->kij",candidate_basis,torch.stack(cleaned),candidate_basis).detach(); base=torch.from_numpy(frontier["rank_8"]).cuda().float(); programs={}
    for name,state in checkpoint["programs"].items():
        program=RoutedTensorProgram(config["candidate_pool_rank"],config["expert_rank"],config["experts"],config["router_features"],base,config["seed"]).cuda(); program.load_state_dict(state); program.eval(); programs[name]=program
    atlas={}; cross_data={}
    for corpus,spec in config["atlas"].items():
        print("atlas",corpus,flush=True); raw_source=np.load(spec["path"],mmap_mode="r"); rows=np.asarray(raw_source[spec["start"]:spec["start"]+spec["rows"]]); states,_=collect(model,config["layer"],spec,config); coordinates=(states@inverse)@candidate_basis; positions=np.asarray(config["positions"]); current=rows[:,positions].reshape(-1); previous=rows[:,positions-1].reshape(-1); next_token=rows[:,positions+1].reshape(-1); position=np.tile(np.arange(len(positions)),len(rows)); current_shape=all_shape[current]; next_shape=all_shape[next_token]; current_flags=all_flags[current]; next_flags=all_flags[next_token]; current_mask=structural_bitmask(current_flags); next_mask=structural_bitmask(next_flags); row_ids=np.repeat(np.arange(len(rows)),len(positions)); train=row_ids%2==0; test=~train
        corpus_rows_result={"variables":{},"seeds":{},"pairwise_ari":{}}
        labels_by_seed={}; profiles={}
        for name,program in programs.items():
            with torch.no_grad(): prediction,gates=program(coordinates,forms,hard=False); labels=gates.argmax(1).cpu().numpy(); confidence=gates.max(1).values.cpu().numpy()
            labels_by_seed[name]=labels; variables={"position":position,"current_token":current,"previous_token":previous,"next_token":next_token,"current_shape":current_shape,"next_shape":next_shape,"current_structure":current_mask,"next_structure":next_mask,"previous_current_bigram":previous.astype(np.int64)*tokenizer.vocab_size+current}
            predictability={}
            for variable,values in variables.items():
                mapping,default=lookup_fit(values[train],labels[train]); predictability[variable]={"heldout":lookup_score(values[test],labels[test],mapping,default),"mapping":mapping if variable in ("position","current_shape","current_structure") else None}
            profiles[name]=semantic_profile(labels,current_shape,next_shape,current_flags,next_flags,position,config["experts"])
            expert_rows=[]
            for expert in range(config["experts"]):
                mean_score=prediction[torch.as_tensor(labels==expert,device="cuda")].mean(0).cpu().tolist()
                expert_rows.append({"share":float(np.mean(labels==expert)),"current_tokens":top_token_lift(current,labels,expert,tokenizer),"next_tokens":top_token_lift(next_token,labels,expert,tokenizer),"top_contexts":top_contexts(rows,positions,labels,confidence,expert,tokenizer),"output_role":output_role(model,mean_score,output_basis,tokenizer),"mean_score":mean_score})
            corpus_rows_result["seeds"][name]={"predictability":predictability,"experts":expert_rows}
        names=list(programs)
        for first,second in itertools.combinations(names,2): corpus_rows_result["pairwise_ari"][f"{first}__{second}"]=adjusted_rand(labels_by_seed[first],labels_by_seed[second])
        corpus_rows_result["profile_alignment"]={name:align_profiles(profiles["seed_0"],profiles[name]) for name in names if name!="seed_0"}
        atlas[corpus]=corpus_rows_result
        cross_data[corpus]={"current":current,"labels":labels_by_seed}
    cross_corpus={}
    for name in programs:
        mapping,default=lookup_fit(cross_data["fineweb"]["current"],cross_data["fineweb"]["labels"][name]); cross_corpus[name]={}
        for corpus in config["atlas"]:
            if corpus=="fineweb":continue
            cross_corpus[name][corpus]=lookup_score(cross_data[corpus]["current"],cross_data[corpus]["labels"][name],mapping,default)
    e1=all(atlas["fineweb"]["seeds"][name]["predictability"]["current_token"]["heldout"]["accuracy"]>=.45 for name in programs); e2=all(row[corpus]["accuracy"]>=.35 for row in cross_corpus.values() for corpus in ("wikitext","tinystories")); e3=all(atlas["fineweb"]["seeds"][name]["predictability"]["position"]["heldout"]["accuracy"]<=.35 for name in programs); e4=all(atlas["fineweb"]["seeds"][name]["predictability"]["current_shape"]["heldout"]["accuracy"]>=.30 for name in programs); e5=all(row["mean_cosine"]>=.8 for row in atlas["fineweb"]["profile_alignment"].values()); e6=all(value>=.2 for value in atlas["fineweb"]["pairwise_ari"].values())
    hypotheses={"E1_current_token_predicts_routes":{"supported":e1},"E2_lexical_routes_transfer_ood":{"supported":e2},"E3_routes_are_not_position_lookup":{"supported":e3},"E4_structural_shape_has_signal":{"supported":e4},"E5_semantic_profiles_recur":{"supported":e5},"E6_route_partitions_recur_pairwise":{"supported":e6}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"cross_corpus":cross_corpus,"atlas":atlas},indent=2,allow_nan=False))
    lines=["# Semantic atlas of the routed Layer-17 program","",f"The atlas uses {config['atlas']['fineweb']['rows']*len(config['positions']):,} disjoint FineWeb sites and {config['atlas']['wikitext']['rows']*len(config['positions']):,} sites from each OOD corpus. Route labels are audited against token identity, structural token shape, position, adjacent tokens, seed-stable semantic profiles, weight-derived output roles, and decoded high-confidence contexts.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Semantic atlas](layer17_routed_semantic_atlas.png)","","| Seed | Fine current-token | Fine current-shape | Fine position | Wiki lexical transfer | Tiny lexical transfer |","|---|---:|---:|---:|---:|---:|"]
    for name in programs:
        p=atlas["fineweb"]["seeds"][name]["predictability"]; lines.append(f"| {name} | {p['current_token']['heldout']['accuracy']:.3f} | {p['current_shape']['heldout']['accuracy']:.3f} | {p['position']['heldout']['accuracy']:.3f} | {cross_corpus[name]['wikitext']['accuracy']:.3f} | {cross_corpus[name]['tinystories']['accuracy']:.3f} |")
    lines += ["","### Reference-seed worked examples",""]
    for expert,row in enumerate(atlas["fineweb"]["seeds"]["seed_0"]["experts"]):
        current=", ".join(repr(x["text"]) for x in row["current_tokens"][:4]); promoted=", ".join(repr(x) for x in row["output_role"]["promoted"][:4]); suppressed=", ".join(repr(x) for x in row["output_role"]["suppressed"][:4]); lines += [f"#### Expert {expert}","",f"Share {row['share']:.3f}. Lifted current tokens: {current}. Approximate weight-based output role promotes {promoted}; suppresses {suppressed}.",""]
        for example in row["top_contexts"][:3]:lines.append(f"- `{example['context']}` (current `{example['current']}`, confidence {example['confidence']:.3f})")
        lines.append("")
    lines += ["Exact expert names remain seed-dependent. Recurrent coarse profiles support a semantic routing ontology; failure of profile or ARI gates means examples should be treated as one quantization's implementation details."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(15,4.5)); names=list(programs); variables=["current_token","previous_token","current_shape","position"]
    for variable in variables:plots[0].plot(names,[atlas["fineweb"]["seeds"][n]["predictability"][variable]["heldout"]["accuracy"] for n in names],marker="o",label=variable)
    plots[0].set_title("FineWeb route predictability");plots[0].legend(fontsize=8); plots[1].bar(np.arange(len(names))-.18,[cross_corpus[n]["wikitext"]["accuracy"] for n in names],.36,label="WikiText");plots[1].bar(np.arange(len(names))+.18,[cross_corpus[n]["tinystories"]["accuracy"] for n in names],.36,label="TinyStories");plots[1].set_xticks(np.arange(len(names)),names);plots[1].set_title("FineWeb lexical-map transfer");plots[1].legend(); pairs=list(atlas["fineweb"]["pairwise_ari"]);plots[2].bar(pairs,list(atlas["fineweb"]["pairwise_ari"].values()));plots[2].set_title("Seed-pair route ARI");plots[2].tick_params(axis="x",rotation=30)
    for plot in plots:plot.grid(alpha=.2)
    figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
