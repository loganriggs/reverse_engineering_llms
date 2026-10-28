#!/usr/bin/env python3
"""Select source edges using their cooperative downstream Fisher Gram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_source_fisher_gram_program.json")
OUTPUT = TENSOR_ROOT / "bilin12_source_fisher_gram_program.json"
REPORT = TENSOR_ROOT / "BILIN12_SOURCE_FISHER_GRAM_PROGRAM.md"
FIGURE = TENSOR_ROOT / "bilin12_source_fisher_gram_program.png"


def fisher_center(probability: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return values - (probability[:, None] * values).sum(-1, keepdim=True) if values.ndim == 3 else values - (probability * values).sum(-1, keepdim=True)


def cooperative_order(probability: torch.Tensor, target: torch.Tensor, effects: torch.Tensor):
    """Forward greedy order under the additive-logit Fisher quadratic."""
    target = fisher_center(probability, target.float()); effects = fisher_center(probability, effects.float())
    target_energy = (probability * target.square()).sum(-1).clamp_min(1e-30)
    edge_energy = (probability[:, None] * effects.square()).sum(-1)
    initial_cross = (probability[:, None] * effects * target[:, None]).sum(-1)
    residual = target.clone(); selected = torch.zeros(effects.shape[:2], device=effects.device, dtype=torch.bool); order = []; frontier = []
    batch = torch.arange(len(target), device=effects.device)
    for _ in range(effects.shape[1]):
        cross = (probability[:, None] * effects * residual[:, None]).sum(-1)
        gain = (2 * cross - edge_energy).masked_fill(selected, -torch.inf)
        choice = gain.argmax(-1); order.append(choice); selected[batch, choice] = True; residual = residual - effects[batch, choice]
        frontier.append(1 - (probability * residual.square()).sum(-1) / target_energy)
    order_tensor = torch.stack(order, 1); signs = initial_cross.gather(1, order_tensor)
    return order_tensor, torch.stack(frontier, 1), signs


def empty_metric():
    return {"examples": 0, "target": 0., "prediction": 0., "cross": 0., "error": 0., "forward_full": 0., "reverse_full": 0., "forward": 0., "target_error": 0.}


def finish_metric(row):
    target=max(row["target"],1e-30);prediction=max(row["prediction"],1e-30)
    return {"examples":row["examples"],"fisher_fidelity":1-row["error"]/target,"fisher_projection":row["cross"]/target,"fisher_cosine":row["cross"]/max(np.sqrt(target*prediction),1e-30),"forward_kl_ratio":row["forward"]/max(row["forward_full"],1e-30),"target_error_kl_ratio":row["target_error"]/max(row["reverse_full"],1e-30)}


@torch.inference_mode()
def evaluate_condition(model, tokenizer, rows, edit_position, config, random_seed):
    newline=tokenizer.encode("\n",add_special_tokens=False)[0];position=config["prediction_position"];keys=[(m,k) for m in config["methods"] for k in config["budgets"]]
    accum={str(layer):{f"{m}/{k}":empty_metric() for m,k in keys} for layer in config["layers"]};diag={str(layer):{"examples":0,"additive_fidelity":[],"linear_greedy":{str(k):[] for k in config["budgets"]},"negative_selected":0,"selected_total":0,"top16_counts":np.zeros(config["sequence_length"],dtype=np.int64)} for layer in config["layers"]};generator=torch.Generator(device="cuda").manual_seed(random_seed)
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:config["sequence_length"]],dtype=np.int64);base_ids=torch.as_tensor(raw,device="cuda",dtype=torch.long);keep=base_ids[:,edit_position]!=newline
        if not bool(keep.any()):continue
        base_ids=base_ids[keep];changed_ids=base_ids.clone();changed_ids[:,edit_position]=newline;base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids)
        for layer in config["layers"]:
            name=str(layer);base=base_rows[layer];changed=changed_rows[layer];block=model.blocks[layer];base_sources=source_write_contributions(block,base["details"],position);changed_sources=source_write_contributions(block,changed["details"],position);source_delta=(changed_sources-base_sources).float()
            full_write=base["details"]["write"].clone();full_write[:,position]=changed["details"]["write"][:,position];base_logits,target_logits=logits_from_attention_writes(model,layer,base,torch.stack((base["details"]["write"],full_write)));target_delta=target_logits-base_logits;probability=F.softmax(base_logits.float(),-1)
            edge_logits=[]
            for chunk_start in range(0,config["sequence_length"],config["world_chunk"]):
                worlds=[]
                for source in range(chunk_start,min(chunk_start+config["world_chunk"],config["sequence_length"])):
                    write=base["details"]["write"].clone();write[:,position]+=source_delta[:,source].to(write.dtype);worlds.append(write)
                edge_logits.append(logits_from_attention_writes(model,layer,base,torch.stack(worlds)))
            edge_effects=torch.cat(edge_logits)-base_logits[None];edge_effects=edge_effects.permute(1,0,2).contiguous()
            greedy,linear_frontier,selected_initial_cross=cooperative_order(probability,target_delta,edge_effects);target_centered=fisher_center(probability,target_delta.float());effects_centered=fisher_center(probability,edge_effects.float());edge_energy=(probability[:,None]*effects_centered.square()).sum(-1);initial_cross=(probability[:,None]*effects_centered*target_centered[:,None]).sum(-1);oneshot=(2*initial_cross-edge_energy).argsort(-1,descending=True);orders={"cooperative_greedy":greedy,"oneshot_gain":oneshot,"absolute_cross":initial_cross.abs().argsort(-1,descending=True),"state_norm":source_delta.square().sum(-1).argsort(-1,descending=True),"random":torch.stack([torch.randperm(config["sequence_length"],generator=generator,device="cuda") for _ in range(len(base_ids))])}
            target_energy=(probability*target_centered.square()).sum(-1).clamp_min(1e-30);additive_residual=target_centered-effects_centered.sum(1);d=diag[name];d["examples"]+=len(base_ids);d["additive_fidelity"].extend((1-(probability*additive_residual.square()).sum(-1)/target_energy).cpu().tolist())
            for budget in config["budgets"]:d["linear_greedy"][str(budget)].extend(linear_frontier[:,budget-1].cpu().tolist())
            first16=greedy[:,:16];d["negative_selected"]+=int((selected_initial_cross[:,:16]<0).sum());d["selected_total"]+=first16.numel()
            for batch_index in range(len(base_ids)):np.add.at(d["top16_counts"],first16[batch_index].cpu().tolist(),1)
            approximations=[];labels=[]
            for method,budget in keys:
                indices=orders[method][:,:budget];mask=torch.zeros(len(base_ids),config["sequence_length"],device="cuda");mask.scatter_(1,indices,1);approximations.append((mask[:,:,None]*source_delta).sum(1));labels.append(f"{method}/{budget}")
            worlds=[]
            for approximation in approximations:
                write=base["details"]["write"].clone();write[:,position]+=approximation.to(write.dtype);worlds.append(write)
            logits=logits_from_attention_writes(model,layer,base,torch.stack(worlds));te,_,_,_=fisher_terms(base_logits,target_delta,target_delta);forward_full=exact_kl(base_logits,target_logits);reverse_full=exact_kl(target_logits,base_logits)
            for index,label in enumerate(labels):
                row=accum[name][label];prediction_delta=logits[index]-base_logits;_,pe,cross,error=fisher_terms(base_logits,target_delta,prediction_delta);row["examples"]+=len(base_ids);row["target"]+=float(te.sum());row["prediction"]+=float(pe.sum());row["cross"]+=float(cross.sum());row["error"]+=float(error.sum());row["forward_full"]+=float(forward_full.sum());row["reverse_full"]+=float(reverse_full.sum());row["forward"]+=float(exact_kl(base_logits,logits[index]).sum());row["target_error"]+=float(exact_kl(target_logits,logits[index]).sum())
    return {"metrics":{layer:{key:finish_metric(row) for key,row in rows.items()} for layer,rows in accum.items()},"diagnostics":{layer:{**{k:v for k,v in row.items() if k!="top16_counts"},"top16_counts":row["top16_counts"].tolist()} for layer,row in diag.items()}}


def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");evaluation={}
    for corpus_index,(corpus,spec) in enumerate(config["evaluation"].items()):
        rows=load_rows(spec);evaluation[corpus]={}
        for position in config["edit_positions"]:print("evaluating",corpus,position,flush=True);evaluation[corpus][str(position)]=evaluate_condition(model,tokenizer,rows,position,config,config["seed"]+1000*corpus_index+position)
    def vals(method,budget):return [evaluation[c][str(p)]["metrics"][str(layer)][f"{method}/{budget}"]["fisher_fidelity"] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    diagnostics=[row for c in evaluation.values() for pos in c.values() for row in pos["diagnostics"].values()];additive=[value for row in diagnostics for value in row["additive_fidelity"]];negative_fraction=sum(row["negative_selected"] for row in diagnostics)/sum(row["selected_total"] for row in diagnostics);counts=np.sum([np.asarray(row["top16_counts"]) for row in diagnostics],axis=0);skeleton=float(np.sort(counts)[-16:].sum()/counts.sum());linear=[];finite=[]
    for c in evaluation:
        for p in config["edit_positions"]:
            for layer in config["layers"]:
                d=evaluation[c][str(p)]["diagnostics"][str(layer)]
                for budget in config["budgets"][:-1]:linear.append(float(np.mean(d["linear_greedy"][str(budget)])));finite.append(evaluation[c][str(p)]["metrics"][str(layer)][f"cooperative_greedy/{budget}"]["fisher_fidelity"])
    correlation=float(np.corrcoef(linear,finite)[0,1]);frontier=[float(np.mean(vals("cooperative_greedy",budget))) for budget in config["budgets"]]
    hypotheses={"G1_individual_edge_effects_are_approximately_additive":{"supported":np.mean(additive)>=.90,"mean":float(np.mean(additive))},"G2_sixteen_cooperative_edges_are_useful":{"supported":np.mean(vals("cooperative_greedy",16))>=.80},"G3_thirtytwo_cooperative_edges_are_robust":{"supported":min(vals("cooperative_greedy",32))>=.75,"minimum":float(min(vals("cooperative_greedy",32)))},"G4_cooperative_selection_beats_state_norm":{"supported":np.mean(vals("cooperative_greedy",16))-np.mean(vals("state_norm",16))>=.10,"mean_advantage":float(np.mean(vals("cooperative_greedy",16))-np.mean(vals("state_norm",16)))},"G5_cooperative_selection_beats_oneshot_gain":{"supported":np.mean(vals("cooperative_greedy",16))-np.mean(vals("oneshot_gain",16))>=.05,"mean_advantage":float(np.mean(vals("cooperative_greedy",16))-np.mean(vals("oneshot_gain",16)))},"G6_cooperative_frontier_is_monotone":{"supported":all(b>=a-1e-6 for a,b in zip(frontier[:-1],frontier[1:]))},"G7_linear_gram_prediction_tracks_finite_behavior":{"supported":correlation>=.90,"correlation":correlation},"G8_cooperative_program_uses_both_signs":{"supported":negative_fraction>=.10,"negative_fraction":negative_fraction},"G9_thirtytwo_edges_are_nearly_sufficient_on_average":{"supported":np.mean(vals("cooperative_greedy",32))>=.90},"G10_no_fixed_source_skeleton_explains_the_program":{"supported":skeleton<.50,"fraction":skeleton,"top_positions":np.argsort(counts)[-16:][::-1].tolist()}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"evaluation":evaluation,"global_top16_counts":counts.tolist()},indent=2,allow_nan=False))
    budgets=config["budgets"];fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for method in config["methods"]:axes[0].plot(budgets,[np.mean(vals(method,b)) for b in budgets],marker="o",label=method);axes[1].plot(budgets,[min(vals(method,b)) for b in budgets],marker="o",label=method)
    axes[0].set_xscale("log",base=2);axes[1].set_xscale("log",base=2);axes[0].set_title("Exact finite mean");axes[1].set_title("Exact finite minimum");axes[2].scatter(linear,finite,s=12,alpha=.7);axes[2].plot([min(linear),max(linear)],[min(linear),max(linear)],color="black",ls="--",lw=.8);axes[2].set_title("Linear Gram versus finite");axes[2].set_xlabel("linear fidelity");axes[2].set_ylabel("finite fidelity")
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=6) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xlabel("source budget");axes[1].set_xlabel("source budget");axes[0].set_ylabel("Fisher fidelity");axes[1].set_ylabel("Fisher fidelity");fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Cooperative downstream-Fisher source program","","For each example, layer, and source position, the finite logit effect of inserting that one exact tensor edge is measured. These effects define an implicit downstream Fisher Gram. Forward selection repeatedly chooses the source edge that most reduces the current additive-logit residual, accounting for self-energy and overlap with already selected edges. Frozen state-norm, one-shot gain, absolute-cross, and random rankings are controls.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Fisher Gram program](bilin12_source_fisher_gram_program.png)","","## Exact finite frontier","","| Budget | Cooperative mean/min | One-shot gain mean/min | Absolute cross mean/min | State norm mean/min |","|---:|---:|---:|---:|---:|"]
    for budget in budgets:lines.append(f"| {budget} | {np.mean(vals('cooperative_greedy',budget)):.3f}/{min(vals('cooperative_greedy',budget)):.3f} | {np.mean(vals('oneshot_gain',budget)):.3f}/{min(vals('oneshot_gain',budget)):.3f} | {np.mean(vals('absolute_cross',budget)):.3f}/{min(vals('absolute_cross',budget)):.3f} | {np.mean(vals('state_norm',budget)):.3f}/{min(vals('state_norm',budget)):.3f} |")
    lines += ["",f"Mean all-edge additive-logit fidelity: **{np.mean(additive):.3f}**. Linear-Gram versus exact-finite correlation: **{correlation:.3f}**. Negative-initial-cross share among the first 16 cooperative selections: **{negative_fraction:.3f}**. Global 16-position skeleton share: **{skeleton:.3f}**.","","This is an adaptive oracle program, not a deployable fixed circuit. It tests whether the cancellation web has a sparse cooperative description once the correct quadratic metric is used."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
