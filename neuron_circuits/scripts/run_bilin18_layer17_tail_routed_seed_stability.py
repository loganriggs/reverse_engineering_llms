#!/usr/bin/env python3
"""Seed stability of the winning four-by-eight routed tail program."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram, evaluate, fit_fisher
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer17_tail_routed_seed_stability.json")
SOURCE_ARRAYS=TENSOR_ROOT/"layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS=TENSOR_ROOT/"layer17_tail_continuous_rank_frontier.npz"
GRANULARITY_CHECKPOINT=TENSOR_ROOT/"layer17_tail_routed_granularity_sweep.pt"
OUTPUT=TENSOR_ROOT/"layer17_tail_routed_seed_stability.json"
CHECKPOINT=TENSOR_ROOT/"layer17_tail_routed_seed_stability.pt"
REPORT=TENSOR_ROOT/"LAYER17_TAIL_ROUTED_SEED_STABILITY.md"
FIGURE=TENSOR_ROOT/"layer17_tail_routed_seed_stability.png"


def geometric_mean(values):
    return float(math.exp(sum(math.log(max(float(v),1e-30)) for v in values)/len(values)))


def best_alignment(reference,candidate):
    count=reference.shape[0]; overlap=torch.empty(count,count,device=reference.device)
    for i in range(count):
        for j in range(count): overlap[i,j]=torch.linalg.svdvals(reference[i].T@candidate[j]).mean()
    best=None
    for permutation in itertools.permutations(range(count)):
        score=float(torch.stack([overlap[i,permutation[i]] for i in range(count)]).mean())
        if best is None or score>best[0]: best=(score,permutation)
    score,permutation=best; inverse=[0]*count
    for reference_index,candidate_index in enumerate(permutation): inverse[candidate_index]=reference_index
    return {"mean_overlap":score,"reference_to_candidate":list(permutation),"candidate_to_reference":inverse,"overlap_matrix":overlap.cpu().tolist()}


@torch.inference_mode()
def hard_routes(program,coordinates):
    logits=coordinates[:,:program.router_features]@program.router_weight+program.router_bias
    return logits.argmax(-1)


def main():
    config=json.loads(CONFIG.read_text()); source=np.load(SOURCE_ARRAYS); frontier=np.load(FRONTIER_ARRAYS); prior=torch.load(GRANULARITY_CHECKPOINT,map_location="cuda",weights_only=False)
    output_basis=torch.from_numpy(source["tail_output_basis"][:,:4]).cuda().float(); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    native=model.blocks[config["layer"]].mlp; down,left,right=native.Down.weight.float().detach(),native.Left.weight.float().detach(),native.Right.weight.float().detach()
    print("constructing frozen objective",flush=True); discovery_x,discovery_write=collect(model,config["layer"],config["discovery"],config); square,inverse,covariance=covariance_transforms(discovery_x,config["covariance_floor_fraction"])
    cleaned,traces=[],[]
    for component in range(4):
        coefficients=down.T@output_basis[:,component]; raw=left.T@(coefficients[:,None]*right); raw=.5*(raw+raw.T); clean,trace=trace_clean(square@raw@square); cleaned.append(clean); traces.append(trace)
    traces_tensor=torch.stack(traces); _,common_basis=common_input(cleaned); candidate_basis=common_basis[:,:config["candidate_pool_rank"]]; forms=torch.einsum("di,kde,ej->kij",candidate_basis,torch.stack(cleaned),candidate_basis).detach()
    z=discovery_x@inverse; coordinates=(z@candidate_basis).detach(); isotropic=(z.square().sum(1,keepdim=True)*traces_tensor[None]).detach(); target=(discovery_write@output_basis-isotropic).detach(); metric=collect_fisher(model,config["discovery"],config,output_basis).detach()
    row_ids=torch.arange(len(discovery_x),device="cuda")//len(config["positions"]); even=torch.nonzero(row_ids%2==0,as_tuple=False).flatten(); odd=torch.nonzero(row_ids%2==1,as_tuple=False).flatten(); base=torch.from_numpy(frontier["rank_8"]).cuda().float(); global32=torch.from_numpy(frontier["rank_32"]).cuda().float()
    programs={}; histories={}; opposite_half_fisher={}
    reference=RoutedTensorProgram(config["candidate_pool_rank"],config["expert_rank"],config["experts"],config["router_features"],base,config["seed"]).cuda(); reference.load_state_dict(prior["programs"]["e4_r8"]); programs["seed_0"]=reference
    for replicate in range(1,config["replicates"]):
        name=f"seed_{replicate}"; print("training",name,flush=True); program=RoutedTensorProgram(config["candidate_pool_rank"],config["expert_rank"],config["experts"],config["router_features"],base,config["seed"]+replicate).cuda(); local={**config,"seed":config["seed"]+100*replicate}; histories[name]=fit_fisher(program,coordinates,forms,target,metric,even,odd,local); programs[name]=program
    for name,program in programs.items():
        with torch.no_grad(): prediction,_=program(coordinates[odd],forms); opposite_half_fisher[name]=float(fisher_quadratic(target[odd]-prediction,metric[odd]).mean())
    results={}; route_labels={corpus:{} for corpus in config["evaluation"]}
    for corpus,spec in config["evaluation"].items():
        print("evaluating",corpus,flush=True); results[corpus]=evaluate(model,spec,config,output_basis,inverse,candidate_basis,forms,traces_tensor,{"global32":global32},programs)
        states,_=collect(model,config["layer"],spec,config); eval_coordinates=(states@inverse)@candidate_basis
        for name,program in programs.items(): route_labels[corpus][name]=hard_routes(program,eval_coordinates)
    alignments={}; route_agreement={corpus:{} for corpus in results}; reference_q=programs["seed_0"].quotients().detach()
    for replicate in range(1,config["replicates"]):
        name=f"seed_{replicate}"; alignment=best_alignment(reference_q,programs[name].quotients().detach()); alignments[name]=alignment; inverse_map=torch.tensor(alignment["candidate_to_reference"],device="cuda")
        for corpus in results: route_agreement[corpus][name]=float((route_labels[corpus]["seed_0"]==inverse_map[route_labels[corpus][name]]).float().mean())
    hard_geomean={name:geometric_mean([r["behavioral"][name+"_hard"]["mean_exact_kl"] for r in results.values()]) for name in programs}; values=np.asarray(list(hard_geomean.values())); cv=float(values.std()/values.mean())
    s1=cv<=.10; s2=all(r["behavioral"][name+"_hard"]["mean_exact_kl"]<=.8*r["behavioral"]["global32"]["mean_exact_kl"] for r in results.values() for name in programs); s3=all(a["mean_overlap"]>=.75 for a in alignments.values()); s4=all(v>=.70 for v in route_agreement["fineweb"].values()); s5=all(v>=.60 for corpus in ("wikitext","tinystories") for v in route_agreement[corpus].values()); s6=s1 and not(s3 and s4 and s5)
    hypotheses={"S1_performance_replicates":{"supported":s1,"cv":cv},"S2_every_seed_beats_global32":{"supported":s2},"S3_expert_subspaces_recur":{"supported":s3},"S4_routes_recur_in_distribution":{"supported":s4},"S5_routes_recur_ood":{"supported":s5},"S6_function_more_stable_than_coordinates":{"supported":s6}}
    torch.save({"config":config,"programs":{n:p.state_dict() for n,p in programs.items()}},CHECKPOINT); OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"opposite_half_fisher":opposite_half_fisher,"hard_geometric_mean_kl":hard_geomean,"performance_cv":cv,"alignments":alignments,"route_agreement":route_agreement,"histories":histories,"results":results},indent=2,allow_nan=False))
    lines=["# Seed stability of the four-by-eight routed tail program","","Four independently initialized programs use the same frozen tensor pool, data split, compute, and balancing objective. Experts are aligned by exhaustive permutation maximizing mean subspace canonical cosine; hard-route agreement is measured only after that alignment.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"Hard-KL geometric-mean coefficient of variation: **{cv:.3f}**.","","| Replicate | Held-out Fisher | Geometric mean KL | Expert overlap to seed 0 | FineWeb route agreement | WikiText | TinyStories |","|---|---:|---:|---:|---:|---:|---:|"]
    for replicate in range(config["replicates"]):
        name=f"seed_{replicate}"; a=1 if replicate==0 else alignments[name]["mean_overlap"]; agreements={c:(1 if replicate==0 else route_agreement[c][name]) for c in results}; lines.append(f"| {name} | {opposite_half_fisher[name]:.4e} | {hard_geomean[name]:.5f} | {a:.3f} | {agreements['fineweb']:.3f} | {agreements['wikitext']:.3f} | {agreements['tinystories']:.3f} |")
    lines += ["","![Seed stability](layer17_tail_routed_seed_stability.png)","","Stable performance with unstable aligned experts would make conditional compactness canonical only at the function class level. Stable expert subspaces and routes would support reusable native-like conditional circuit identities."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(14,4.2)); names=list(programs); plots[0].bar(names,[hard_geomean[n] for n in names]); plots[0].set_title("Hard-route geometric mean KL"); plots[1].bar(list(alignments),[a["mean_overlap"] for a in alignments.values()]); plots[1].set_title("Aligned expert overlap")
    for corpus,row in route_agreement.items(): plots[2].plot(list(row),list(row.values()),marker="o",label=corpus)
    plots[2].set_title("Aligned route agreement"); plots[2].legend();
    for plot in plots: plot.tick_params(axis="x",rotation=25); plot.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(FIGURE,dpi=180); plt.close(figure); print("wrote",OUTPUT,CHECKPOINT,REPORT,FIGURE,flush=True)


if __name__=="__main__": main()
