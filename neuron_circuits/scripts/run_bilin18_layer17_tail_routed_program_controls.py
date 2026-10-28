#!/usr/bin/env python3
"""Specificity controls for the context-routed Layer-17 tail program."""

from __future__ import annotations

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
from run_bilin18_layer17_tail_continuous_fisher import continuous_prediction, score_summary, summarize
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_routed_program_controls.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.npz"
ROUTED_CHECKPOINT = TENSOR_ROOT / "layer17_tail_routed_exact_program.pt"
OUTPUT = TENSOR_ROOT / "layer17_tail_routed_program_controls.json"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_ROUTED_PROGRAM_CONTROLS.md"
FIGURE = TENSOR_ROOT / "layer17_tail_routed_program_controls.png"


def canonical_overlap_matrix(quotients):
    count = quotients.shape[0]
    result = torch.empty(count, count, device=quotients.device)
    for first in range(count):
        for second in range(count):
            result[first, second] = torch.linalg.svdvals(
                quotients[first].T @ quotients[second]
            ).mean()
    return result


def expert_scores(coordinates, forms, quotients):
    reduced_coordinates = torch.einsum("bp,epr->ber", coordinates, quotients)
    reduced_forms = torch.einsum("epa,kpq,eqb->ekab", quotients, forms, quotients)
    return torch.einsum("bei,ekij,bej->bek", reduced_coordinates, reduced_forms, reduced_coordinates)


@torch.inference_mode()
def evaluate(model, spec, config, output_basis, inverse, candidate_basis, forms,
             traces, program, global32, global_mean, position_mean, seed):
    rows = corpus_rows(spec); positions = config["positions"]
    native_block = model.blocks[config["layer"]]; dimension = model.config.n_embd
    names = ["global32", "native_soft", "native_hard", "shuffled_hard", "uniform",
             "global_mean", "position_mean", "position_hard"] + [f"fixed_{i}" for i in range(config["experts"])]
    records = {name: {field: [] for field in ("exact_kl", "ce_delta", "top1")} for name in names}
    predictions_all = {name: [] for name in names}; targets_all = []
    generator = torch.Generator(device="cuda").manual_seed(seed)
    quotients = program.quotients()
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda"); pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1); x = normalized[:, positions].float().flatten(0, 1)
        raw_write = native_block.mlp.Down(native_block.mlp.Left(x) * native_block.mlp.Right(x)).float()
        native_write = raw_write + native_block.mlp.Down_bias.float(); native_score = raw_write @ output_basis
        z = x @ inverse; coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        scores = expert_scores(coordinates, forms, quotients)
        router_logits = coordinates[:, :program.router_features] @ program.router_weight + program.router_bias
        soft = router_logits.softmax(-1); hard = F.one_hot(soft.argmax(-1), config["experts"]).float()
        shuffled = hard[torch.randperm(len(hard), generator=generator, device="cuda")]
        uniform = torch.full_like(soft, 1/config["experts"])
        position_indices = torch.arange(len(hard), device="cuda") % len(positions)
        positional = position_mean[position_indices]
        positional_hard = F.one_hot(positional.argmax(-1), config["experts"]).float()
        gates = {
            "native_soft": soft, "native_hard": hard, "shuffled_hard": shuffled,
            "uniform": uniform, "global_mean": global_mean.expand_as(soft),
            "position_mean": positional, "position_hard": positional_hard,
        }
        predictions = {
            "global32": continuous_prediction(coordinates, forms, global32) + isotropic,
            **{name: torch.einsum("be,bek->bk", gate, scores) + isotropic for name, gate in gates.items()},
            **{f"fixed_{i}": scores[:, i] + isotropic for i in range(config["experts"])},
        }
        targets_all.append(native_score.cpu())
        native_logits = 30*torch.tanh(model.lm_head(F.rms_norm(pre+native_write,(dimension,))).float()/30)
        native_logprob = native_logits.log_softmax(-1); native_probability = native_logprob.exp(); native_top1=native_logits.argmax(-1)
        token_targets = torch.as_tensor(rows[start:start+len(raw),np.asarray(positions)+1].reshape(-1),device="cuda",dtype=torch.long)
        native_ce = -native_logprob.gather(1,token_targets[:,None]).squeeze(1)
        for name,prediction in predictions.items():
            predictions_all[name].append(prediction.cpu()); changed=native_write+(prediction-native_score)@output_basis.T
            logits=30*torch.tanh(model.lm_head(F.rms_norm(pre+changed,(dimension,))).float()/30); logprob=logits.log_softmax(-1)
            kl=(native_probability*(native_logprob-logprob)).sum(1); ce=-logprob.gather(1,token_targets[:,None]).squeeze(1)
            records[name]["exact_kl"].extend(kl.cpu().tolist()); records[name]["ce_delta"].extend((ce-native_ce).cpu().tolist()); records[name]["top1"].extend((logits.argmax(-1)==native_top1).cpu().tolist())
    target=torch.cat(targets_all)
    return {"scores":{n:score_summary(target,torch.cat(v)) for n,v in predictions_all.items()},"behavioral":{n:summarize(v) for n,v in records.items()}}


def main():
    config=json.loads(CONFIG.read_text()); source=np.load(SOURCE_ARRAYS); frontier=np.load(FRONTIER_ARRAYS); checkpoint=torch.load(ROUTED_CHECKPOINT,map_location="cuda",weights_only=False)
    output_basis=torch.from_numpy(source["tail_output_basis"][:,:4]).cuda().float(); model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    native=model.blocks[config["layer"]].mlp; down,left,right=native.Down.weight.float().detach(),native.Left.weight.float().detach(),native.Right.weight.float().detach()
    discovery_x,_=collect(model,config["layer"],config["discovery"],config); square,inverse,covariance=covariance_transforms(discovery_x,config["covariance_floor_fraction"])
    cleaned,traces=[],[]
    for component in range(4):
        coefficients=down.T@output_basis[:,component]; raw=left.T@(coefficients[:,None]*right); raw=.5*(raw+raw.T); clean,trace=trace_clean(square@raw@square); cleaned.append(clean); traces.append(trace)
    traces_tensor=torch.stack(traces); _,common_basis=common_input(cleaned); candidate_basis=common_basis[:,:config["candidate_pool_rank"]]; forms=torch.einsum("di,kde,ej->kij",candidate_basis,torch.stack(cleaned),candidate_basis)
    global8=torch.from_numpy(frontier["rank_8"]).cuda().float(); global32=torch.from_numpy(frontier["rank_32"]).cuda().float()
    program=RoutedTensorProgram(config["candidate_pool_rank"],config["expert_rank"],config["experts"],config["router_features"],global8,config["seed"]).cuda(); program.load_state_dict(checkpoint["fisher"]); program.eval()
    with torch.no_grad():
        z=discovery_x@inverse; coordinates=z@candidate_basis; _,gates=program(coordinates,forms)
        global_mean=gates.mean(0); position_mean=gates.reshape(config["discovery"]["rows"],len(config["positions"]),config["experts"]).mean(0)
        overlaps=canonical_overlap_matrix(program.quotients())
    results={}
    for index,(corpus,spec) in enumerate(config["evaluation"].items()):
        print("evaluating controls",corpus,flush=True); results[corpus]=evaluate(model,spec,config,output_basis,inverse,candidate_basis,forms,traces_tensor,program,global32,global_mean,position_mean,config["seed"]+index)
    d1=all(r["behavioral"]["shuffled_hard"]["mean_exact_kl"]>=1.25*r["behavioral"]["native_hard"]["mean_exact_kl"] for r in results.values())
    d2=all(r["behavioral"]["native_hard"]["mean_exact_kl"]<=.8*min(r["behavioral"]["position_mean"]["mean_exact_kl"],r["behavioral"]["position_hard"]["mean_exact_kl"]) for r in results.values())
    d3=all(min(r["behavioral"][f"fixed_{i}"]["mean_exact_kl"] for i in range(config["experts"]))>=1.1*r["behavioral"]["native_hard"]["mean_exact_kl"] for r in results.values())
    d4=all(float(overlaps[i,j])<.9 for i in range(config["experts"]) for j in range(i+1,config["experts"]))
    d5=all(r["behavioral"]["native_hard"]["mean_exact_kl"]<=.8*r["behavioral"]["global32"]["mean_exact_kl"] for r in results.values())
    d6=all(r["behavioral"]["uniform"]["mean_exact_kl"]>r["behavioral"]["native_hard"]["mean_exact_kl"] for r in results.values())
    hypotheses={"D1_context_assignment_matters":{"supported":d1},"D2_context_beats_position_only":{"supported":d2},"D3_no_single_expert_matches_router":{"supported":d3},"D4_experts_are_distinct_subspaces":{"supported":d4},"D5_hard_router_beats_global32":{"supported":d5},"D6_uniform_ensemble_is_insufficient":{"supported":d6}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"global_mean_gate":global_mean.cpu().tolist(),"position_mean_gate":position_mean.cpu().tolist(),"expert_overlap":overlaps.cpu().tolist(),"results":results},indent=2,allow_nan=False))
    lines=["# Specificity controls for the routed Layer-17 tail program","","The Fisher-trained four-by-eight program is frozen. Route assignments are shuffled, replaced by global or position-only means, made uniform, or removed by running each expert alone. All controls retain the same experts and exact final-logit intervention.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    off=[float(overlaps[i,j]) for i in range(config["experts"]) for j in range(i+1,config["experts"])]; lines += ["",f"Expert-subspace mean canonical cosine: **{np.mean(off):.3f}** (range {np.min(off):.3f}--{np.max(off):.3f}).","","![Route controls](layer17_tail_routed_program_controls.png)","","| Corpus | Program | Mean KL | Median KL | CE Δ | Top1 |","|---|---|---:|---:|---:|---:|"]
    display=["global32","native_hard","shuffled_hard","uniform","global_mean","position_mean","position_hard"]+[f"fixed_{i}" for i in range(config["experts"])]
    for corpus,row in results.items():
        for name in display:
            b=row["behavioral"][name]; lines.append(f"| {corpus} | {name} | {b['mean_exact_kl']:.5f} | {b['median_exact_kl']:.5f} | {b['ce_delta']:.5f} | {b['top1_agreement']:.3f} |")
    lines += ["","If shuffled, position-only, and fixed-expert controls lose the gain, the result is conditional circuit selection. If they match it, routing is merely an ensemble or positional lookup."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure,plots=plt.subplots(1,2,figsize=(14,4.8)); names=["global32","native_hard","shuffled_hard","uniform","position_mean"]+[f"fixed_{i}" for i in range(config["experts"])]
    for corpus,row in results.items(): plots[0].plot(names,[row["behavioral"][n]["mean_exact_kl"] for n in names],marker="o",label=corpus)
    image=plots[1].imshow(overlaps.cpu(),vmin=0,vmax=1,cmap="viridis"); plots[1].set_xticks(range(config["experts"])); plots[1].set_yticks(range(config["experts"])); plots[1].set_title("Expert subspace overlap"); figure.colorbar(image,ax=plots[1])
    plots[0].set_title("Route-specificity controls"); plots[0].set_ylabel("mean exact KL"); plots[0].tick_params(axis="x",rotation=30); plots[0].legend(); plots[0].grid(alpha=.2); figure.tight_layout(); figure.savefig(FIGURE,dpi=180); plt.close(figure); print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__": main()
