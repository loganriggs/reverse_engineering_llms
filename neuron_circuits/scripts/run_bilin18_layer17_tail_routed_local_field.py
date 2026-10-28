#!/usr/bin/env python3
"""Test whether routed programs recover a seed-stable local subspace field."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram
from run_bilin18_layer17_tail_tensor_canonicalization import centered_vector_r2, common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer17_tail_routed_local_field.json")
SOURCE_ARRAYS=TENSOR_ROOT/"layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS=TENSOR_ROOT/"layer17_tail_continuous_rank_frontier.npz"
SEED_CHECKPOINT=TENSOR_ROOT/"layer17_tail_routed_seed_stability.pt"
SEED_RESULTS=TENSOR_ROOT/"layer17_tail_routed_seed_stability.json"
OUTPUT=TENSOR_ROOT/"layer17_tail_routed_local_field.json"
REPORT=TENSOR_ROOT/"LAYER17_TAIL_ROUTED_LOCAL_FIELD.md"
FIGURE=TENSOR_ROOT/"layer17_tail_routed_local_field.png"


def selected_subspace_cosine(first_q,first_labels,second_q,second_labels):
    first=first_q[first_labels]; second=second_q[second_labels]
    products=torch.einsum("bpr,bps->brs",first,second)
    return torch.linalg.svdvals(products).mean(-1)


def route_optimal(first_labels,second_labels,count):
    best=None
    for permutation in itertools.permutations(range(count)):
        inverse=torch.empty(count,dtype=torch.long,device=first_labels.device)
        for first_index,second_index in enumerate(permutation): inverse[second_index]=first_index
        agreement=float((first_labels==inverse[second_labels]).float().mean())
        if best is None or agreement>best[0]: best=(agreement,inverse.tolist())
    return {"agreement":best[0],"candidate_to_reference":best[1]}


@torch.inference_mode()
def exact_output_kl(model,spec,config,output_basis,inverse,candidate_basis,forms,traces,programs):
    rows=corpus_rows(spec); positions=config["positions"]; native=model.blocks[config["layer"]]; dimension=model.config.n_embd
    records={name:[] for name in programs if name!="seed_0"}; score_rows={name:[] for name in programs}; label_rows={name:[] for name in programs}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); ids=torch.as_tensor(raw,device="cuda"); pre,normalized=prefix_to_layer17(model,ids); pre=pre[:,positions].float().flatten(0,1); x=normalized[:,positions].float().flatten(0,1)
        raw_write=native.mlp.Down(native.mlp.Left(x)*native.mlp.Right(x)).float(); native_write=raw_write+native.mlp.Down_bias.float(); native_score=raw_write@output_basis; z=x@inverse; coordinates=z@candidate_basis; isotropic=z.square().sum(1,keepdim=True)*traces[None]
        logprobs={}
        for name,program in programs.items():
            prediction,gates=program(coordinates,forms,hard=True); prediction=prediction+isotropic; score_rows[name].append(prediction.cpu()); label_rows[name].append(gates.argmax(1).cpu()); changed=native_write+(prediction-native_score)@output_basis.T; logits=30*torch.tanh(model.lm_head(F.rms_norm(pre+changed,(dimension,))).float()/30); logprobs[name]=logits.log_softmax(-1)
        reference_probability=logprobs["seed_0"].exp()
        for name in records: records[name].extend((reference_probability*(logprobs["seed_0"]-logprobs[name])).sum(1).cpu().tolist())
    return {"scores":{n:torch.cat(v) for n,v in score_rows.items()},"labels":{n:torch.cat(v).cuda() for n,v in label_rows.items()},"kl":{n:{"mean":float(np.mean(v)),"median":float(np.median(v))} for n,v in records.items()}}


def main():
    config=json.loads(CONFIG.read_text()); source=np.load(SOURCE_ARRAYS); frontier=np.load(FRONTIER_ARRAYS); checkpoint=torch.load(SEED_CHECKPOINT,map_location="cuda",weights_only=False); prior=json.loads(SEED_RESULTS.read_text())
    output_basis=torch.from_numpy(source["tail_output_basis"][:,:4]).cuda().float(); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); native=model.blocks[config["layer"]].mlp; down,left,right=native.Down.weight.float().detach(),native.Left.weight.float().detach(),native.Right.weight.float().detach()
    discovery_x,_=collect(model,config["layer"],config["discovery"],config); square,inverse,covariance=covariance_transforms(discovery_x,config["covariance_floor_fraction"]); cleaned,traces=[],[]
    for component in range(4):
        coefficients=down.T@output_basis[:,component]; raw=left.T@(coefficients[:,None]*right); raw=.5*(raw+raw.T); clean,trace=trace_clean(square@raw@square); cleaned.append(clean); traces.append(trace)
    traces_tensor=torch.stack(traces); _,common_basis=common_input(cleaned); candidate_basis=common_basis[:,:config["candidate_pool_rank"]]; forms=torch.einsum("di,kde,ej->kij",candidate_basis,torch.stack(cleaned),candidate_basis).detach(); base=torch.from_numpy(frontier["rank_8"]).cuda().float()
    programs={}
    for name,state in checkpoint["programs"].items():
        program=RoutedTensorProgram(config["candidate_pool_rank"],config["expert_rank"],config["experts"],config["router_features"],base,config["seed"]).cuda(); program.load_state_dict(state); program.eval(); programs[name]=program
    results={}; generator=torch.Generator(device="cuda").manual_seed(config["seed"])
    for corpus,spec in config["evaluation"].items():
        print("local field",corpus,flush=True); exact=exact_output_kl(model,spec,config,output_basis,inverse,candidate_basis,forms,traces_tensor,programs); reference_labels=exact["labels"]["seed_0"]; reference_q=programs["seed_0"].quotients().detach(); rows={}
        for name,program in programs.items():
            if name=="seed_0": continue
            labels=exact["labels"][name]; q=program.quotients().detach(); observed=selected_subspace_cosine(reference_q,reference_labels,q,labels); route_alignment=route_optimal(reference_labels,labels,config["experts"]); mapping=torch.tensor(route_alignment["candidate_to_reference"],device="cuda"); disagreement=reference_labels!=mapping[labels]
            null_means=[]
            for _ in range(config["null_draws"]):
                shuffled_context=labels[torch.randperm(len(labels),generator=generator,device="cuda")]; null_means.append(float(selected_subspace_cosine(reference_q,reference_labels,q,shuffled_context).mean()))
            rows[name]={"mean_selected_overlap":float(observed.mean()),"median_selected_overlap":float(observed.median()),"disagreement_selected_overlap":float(observed[disagreement].mean()) if disagreement.any() else 1.0,"route_optimal":route_alignment,"shuffled_context_mean":float(np.mean(null_means)),"shuffled_context_p95":float(np.quantile(null_means,.95)),"empirical_p":float((1+sum(v>=float(observed.mean()) for v in null_means))/(1+len(null_means))),"score_r2_to_reference":centered_vector_r2(exact["scores"]["seed_0"].numpy(),exact["scores"][name].numpy()),"exact_output_kl":exact["kl"][name]}
        results[corpus]=rows
    nonref=[n for n in programs if n!="seed_0"]; l1=all(results[c][n]["mean_selected_overlap"]>=.8 for c in results for n in nonref); l2=all(results[c][n]["mean_selected_overlap"]>results[c][n]["shuffled_context_p95"] for c in results for n in nonref); l3=all(results[c][n]["score_r2_to_reference"]>=.8 for c in results for n in nonref); l4=all(results[c][n]["exact_output_kl"]["mean"]<=.002 for c in results for n in nonref); l5=all(results[c][n]["disagreement_selected_overlap"]>=.75 for c in results for n in nonref); prior_ood_failed=not prior["hypotheses"]["S5_routes_recur_ood"]["supported"]; l6=l1 and prior_ood_failed
    hypotheses={"L1_selected_subspace_field_recurs":{"supported":l1},"L2_selected_field_beats_context_shuffle":{"supported":l2},"L3_hard_score_function_recurs":{"supported":l3},"L4_output_function_recurs":{"supported":l4},"L5_disagreement_sites_still_share_subspaces":{"supported":l5},"L6_field_is_more_canonical_than_labels":{"supported":l6}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"results":results},indent=2,allow_nan=False))
    lines=["# Seed-invariant local subspace field","","The discrete expert labels were not seed-stable. This test discards names and compares, at each identical context, the rank-eight subspace actually selected by each seed. Context-shuffled selections preserve each seed's expert usage and form the null. Hard predicted scores and exact output distributions are compared independently.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Local field](layer17_tail_routed_local_field.png)","","| Corpus | Seed | Selected overlap | Shuffled p95 | Disagreement overlap | Route-opt agreement | Score R² | Mean output KL |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for corpus,rows in results.items():
        for name,row in rows.items(): lines.append(f"| {corpus} | {name} | {row['mean_selected_overlap']:.3f} | {row['shuffled_context_p95']:.3f} | {row['disagreement_selected_overlap']:.3f} | {row['route_optimal']['agreement']:.3f} | {row['score_r2_to_reference']:.3f} | {row['exact_output_kl']['mean']:.5f} |")
    lines += ["","If same-context selected subspaces recur while route labels do not, the canonical object is a smooth or overlapping local subspace field; the four experts are a useful quantization, not four unique modules."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,3,figsize=(14,4.3)); x=np.arange(len(nonref)); width=.25
    for ci,(corpus,rows) in enumerate(results.items()): plots[0].bar(x+(ci-1)*width,[rows[n]["mean_selected_overlap"] for n in nonref],width,label=corpus); plots[1].bar(x+(ci-1)*width,[rows[n]["score_r2_to_reference"] for n in nonref],width,label=corpus); plots[2].bar(x+(ci-1)*width,[rows[n]["exact_output_kl"]["mean"] for n in nonref],width,label=corpus)
    for plot,title in zip(plots,("Selected local subspace","Hard score function","Approximation output divergence")): plot.set_xticks(x,nonref); plot.set_title(title); plot.grid(alpha=.2); plot.legend(fontsize=8)
    plots[0].set_ylabel("mean canonical cosine"); plots[1].set_ylabel("centered R²"); plots[2].set_ylabel("mean exact KL"); figure.tight_layout(); figure.savefig(FIGURE,dpi=180); plt.close(figure); print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__": main()
