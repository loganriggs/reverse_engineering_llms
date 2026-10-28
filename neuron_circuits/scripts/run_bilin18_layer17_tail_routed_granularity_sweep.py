#!/usr/bin/env python3
"""Compute-matched conditional-rank sweep for the Layer-17 tail."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import continuous_prediction
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram, evaluate, fit_fisher
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_routed_granularity_sweep.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_routed_granularity_sweep.json"
CHECKPOINT = TENSOR_ROOT / "layer17_tail_routed_granularity_sweep.pt"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_ROUTED_GRANULARITY_SWEEP.md"
FIGURE = TENSOR_ROOT / "layer17_tail_routed_granularity_sweep.png"


def geometric_mean(values):
    return float(math.exp(sum(math.log(max(float(value), 1e-30)) for value in values) / len(values)))


def main():
    config=json.loads(CONFIG.read_text()); source=np.load(SOURCE_ARRAYS); frontier=np.load(FRONTIER_ARRAYS)
    output_basis=torch.from_numpy(source["tail_output_basis"][:,:4]).cuda().float(); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    native=model.blocks[config["layer"]].mlp; down,left,right=native.Down.weight.float().detach(),native.Left.weight.float().detach(),native.Right.weight.float().detach()
    print("constructing frozen tensor objective",flush=True); discovery_x,discovery_write=collect(model,config["layer"],config["discovery"],config); square,inverse,covariance=covariance_transforms(discovery_x,config["covariance_floor_fraction"])
    cleaned,traces=[],[]
    for component in range(4):
        coefficients=down.T@output_basis[:,component]; raw=left.T@(coefficients[:,None]*right); raw=.5*(raw+raw.T); clean,trace=trace_clean(square@raw@square); cleaned.append(clean); traces.append(trace)
    traces_tensor=torch.stack(traces); _,common_basis=common_input(cleaned); candidate_basis=common_basis[:,:config["candidate_pool_rank"]]; forms=torch.einsum("di,kde,ej->kij",candidate_basis,torch.stack(cleaned),candidate_basis).detach()
    z=discovery_x@inverse; coordinates=(z@candidate_basis).detach(); isotropic=(z.square().sum(1,keepdim=True)*traces_tensor[None]).detach(); target=(discovery_write@output_basis-isotropic).detach(); metric=collect_fisher(model,config["discovery"],config,output_basis).detach()
    row_ids=torch.arange(len(discovery_x),device="cuda")//len(config["positions"]); even=torch.nonzero(row_ids%2==0,as_tuple=False).flatten(); odd=torch.nonzero(row_ids%2==1,as_tuple=False).flatten()
    programs={}; histories={}; opposite_half_fisher={}; parameter_count={}
    for index,setting in enumerate(config["settings"]):
        experts,rank=setting["experts"],setting["rank"]; name=f"e{experts}_r{rank}"; print("training",name,flush=True)
        base=torch.from_numpy(frontier[f"rank_{rank}"]).cuda().float(); program=RoutedTensorProgram(config["candidate_pool_rank"],rank,experts,config["router_features"],base,config["seed"]+index).cuda()
        local_config={**config,"seed":config["seed"]+100*index}; histories[name]=fit_fisher(program,coordinates,forms,target,metric,even,odd,local_config)
        with torch.no_grad(): prediction,_=program(coordinates[odd],forms); opposite_half_fisher[name]=float(fisher_quadratic(target[odd]-prediction,metric[odd]).mean())
        programs[name]=program; parameter_count[name]=experts*config["candidate_pool_rank"]*rank+config["router_features"]*experts+experts
    global32=torch.from_numpy(frontier["rank_32"]).cuda().float(); results={}
    for corpus,spec in config["evaluation"].items():
        print("external evaluation",corpus,flush=True); results[corpus]=evaluate(model,spec,config,output_basis,inverse,candidate_basis,forms,traces_tensor,{"global32":global32},programs)
    hard_geomean={name:geometric_mean([r["behavioral"][f"{name}_hard"]["mean_exact_kl"] for r in results.values()]) for name in programs}
    winner=min(hard_geomean,key=hard_geomean.get); winners_by_corpus={corpus:min(programs,key=lambda name:row["behavioral"][f"{name}_hard"]["mean_exact_kl"]) for corpus,row in results.items()}
    baseline="e1_r32"
    g1=any(all(r["behavioral"][f"{name}_hard"]["mean_exact_kl"]<=.8*r["behavioral"][f"{baseline}_hard"]["mean_exact_kl"] for r in results.values()) for name in programs if name!=baseline)
    g2=winner in {"e2_r16","e4_r8","e8_r4"}
    g3=any(all(r["behavioral"][f"{name}_hard"]["median_exact_kl"]<=.005 for r in results.values()) for name in programs if int(name.split("_r")[1])<=4)
    g4=all(r["behavioral"][f"{winner}_hard"]["mean_exact_kl"]<=1.1*r["behavioral"][f"{winner}_soft"]["mean_exact_kl"] for r in results.values())
    g5=len(set(winners_by_corpus.values()))==1
    base_fisher=opposite_half_fisher[baseline]; base_geo=hard_geomean[baseline]
    g6=all(hard_geomean[name]<base_geo for name in programs if name!=baseline and opposite_half_fisher[name]<base_fisher)
    hypotheses={"G1_conditional_program_beats_global32":{"supported":g1},"G2_best_granularity_is_interior":{"supported":g2},"G3_rank4_or_less_can_reach_gate":{"supported":g3},"G4_hard_execution_matches_soft":{"supported":g4},"G5_winner_transfers_across_corpora":{"supported":g5},"G6_conditional_gain_is_not_only_local":{"supported":g6}}
    torch.save({"config":config,"programs":{name:p.state_dict() for name,p in programs.items()}},CHECKPOINT)
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"opposite_half_fisher":opposite_half_fisher,"parameter_count":parameter_count,"hard_geometric_mean_kl":hard_geomean,"winner":winner,"winners_by_corpus":winners_by_corpus,"histories":histories,"results":results},indent=2,allow_nan=False))
    lines=["# Compute-matched routed granularity sweep","","All programs have 32 total expert directions: one rank-32 expert through sixteen rank-2 experts. Only one expert executes under hard routing. Experts and a 32-coordinate linear router are trained on even FineWeb rows with Fisher error and validated on odd rows, then frozen for exact OOD interventions.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"Geometric-mean hard-KL winner: **{winner}**. Per-corpus winners: `{winners_by_corpus}`.","","![Granularity sweep](layer17_tail_routed_granularity_sweep.png)","","| Setting | Parameters | Active rank | Held-out Fisher | Geometric mean KL | "+" | ".join(f"{c} hard/soft KL" for c in results)+" |","|---|---:|---:|---:|---:|"+"---:|"*len(results)]
    for setting in config["settings"]:
        name=f"e{setting['experts']}_r{setting['rank']}"; cells=[f"{results[c]['behavioral'][name+'_hard']['mean_exact_kl']:.5f}/{results[c]['behavioral'][name+'_soft']['mean_exact_kl']:.5f}" for c in results]
        lines.append(f"| {name} | {parameter_count[name]} | {setting['rank']} | {opposite_half_fisher[name]:.4e} | {hard_geomean[name]:.5f} | "+" | ".join(cells)+" |")
    lines += ["","The sweep separates stored parameter budget from simultaneous execution rank. An interior hard-routing optimum supports conditional compactness; an endpoint optimum supports either one broad circuit or very fine discrete lookup."]
    REPORT.write_text("\n".join(lines)+"\n")
    names=list(programs); figure,plots=plt.subplots(1,3,figsize=(15,4.5)); plots[0].plot(names,[opposite_half_fisher[n] for n in names],marker="o"); plots[0].set_title("Held-out Fisher")
    for corpus,row in results.items():
        plots[1].plot(names,[row["behavioral"][n+"_hard"]["mean_exact_kl"] for n in names],marker="o",label=corpus); plots[2].plot(names,[row["behavioral"][n+"_hard"]["median_exact_kl"] for n in names],marker="o",label=corpus)
    plots[1].set_title("Hard-route mean KL"); plots[2].set_title("Hard-route median KL"); plots[2].axhline(.005,color="black",linestyle="--",linewidth=1)
    for plot in plots: plot.tick_params(axis="x",rotation=25); plot.grid(alpha=.2)
    plots[1].legend(); plots[2].legend(); figure.tight_layout(); figure.savefig(FIGURE,dpi=180); plt.close(figure); print("wrote",OUTPUT,CHECKPOINT,REPORT,FIGURE,flush=True)


if __name__=="__main__": main()
