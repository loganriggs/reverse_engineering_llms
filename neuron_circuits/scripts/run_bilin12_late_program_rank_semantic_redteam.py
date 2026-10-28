#!/usr/bin/env python3
"""Red-team late compact programs with stability, rank frontiers, and token edits."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, layer_input, load_snapshot_model
from run_bilin12_late_address_program_audit import (
    build_program, exact_kl, load_rows, run_suffix,
)
from run_bilin12_layer5_weight_conditioned_addresses import (
    derive_cleaned_addresses, output_conditioned_form,
)
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_late_program_rank_semantic_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_late_program_rank_semantic_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LATE_PROGRAM_RANK_SEMANTIC_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_late_program_rank_semantic_redteam.png"


def best_permutation(overlap: np.ndarray) -> tuple[list[int], float]:
    best = None; best_score = -1.0
    for permutation in itertools.permutations(range(overlap.shape[1])):
        score = sum(overlap[i, permutation[i]] for i in range(overlap.shape[0]))
        if score > best_score: best_score = score; best = list(permutation)
    return best, float(best_score / overlap.shape[0])


@torch.inference_mode()
def construct(model, layer, spec, config):
    states, _ = collect_states_writes(model, layer, spec, config["positions"], config["batch_size"])
    square, inverse, covariance = ridge_covariance_transforms(states, config["covariance_ridge_fraction"])
    block = model.blocks[layer]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float()
    _, outputs = output_map_eigensystem(down,left,right); output=outputs[:,0]
    form=output_conditioned_form(down,left,right,output)
    _,_,edits=derive_cleaned_addresses(form,square,config["address_count"])
    programs=[build_program(edits[:,i],output,down,left,right,square,inverse,config["native_prefixes"]) for i in range(config["address_count"])]
    return edits, programs, square, inverse, covariance


@torch.inference_mode()
def newline_substitution(model, tokenizer, layer, programs, config, polarities):
    newline=tokenizer.encode("\n",add_special_tokens=False)
    if len(newline)!=1: raise RuntimeError(f"newline not one token: {newline}")
    newline=newline[0]; result={}
    for corpus,spec in config["substitution_evaluation"].items():
        rows=load_rows(spec); changes=defaultdict(list)
        for position in config["substitution_positions"]:
            for start in range(0,len(rows),config["batch_size"]):
                raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64)
                original=torch.as_tensor(raw,device="cuda"); keep=original[:,position]!=newline
                if not bool(keep.any()): continue
                original=original[keep]; changed=original.clone(); changed[:,position]=newline
                old=layer_input(model,original,layer)[:,position]; new=layer_input(model,changed,layer)[:,position]
                for index,program in enumerate(programs):
                    if str(index) not in polarities: continue
                    delta=(new@program["context_read"]-old@program["context_read"]).cpu().tolist()
                    changes[str(index)].extend(delta)
        result[corpus]={}
        for index,values in changes.items():
            expected=polarities[index]
            array=np.asarray(values); correct=array>0 if expected=="positive" else array<0
            result[corpus][index]={"polarity":expected,"count":len(values),"mean_change":float(array.mean()),"predicted_direction_fraction":float(correct.mean())}
    return result


@torch.inference_mode()
def causal_frontiers(model,layer,programs,inverse,config):
    block=model.blocks[layer]; dimension=model.config.n_embd; result={}
    for corpus,spec in config["causal_evaluation"].items():
        rows=load_rows(spec); store=defaultdict(list)
        for start in range(0,len(rows),config["batch_size"]):
            raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); ids=torch.as_tensor(raw,device="cuda")
            x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; value_bus=None
            for prefix_block in model.blocks[:layer]: x,value_bus=prefix_block(x,value_bus,x0)
            mixed=block.lambdas[0]*x+block.lambdas[1]*x0; attention,value_bus=block.attn(F.rms_norm(mixed,(dimension,)),value_bus)
            pre=mixed+attention; state=F.rms_norm(pre,(dimension,)); final=state[:,-1]
            base_write=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
            native_x=pre+base_write; baseline=run_suffix(model,native_x,value_bus,x0,layer+1)
            whitened=final@inverse
            for index,p in enumerate(programs):
                exact_base=base_write[:,-1]; rank_delta={}
                for rank in config["ranks"]:
                    rank_delta[rank]=(whitened@p["v"][:,:rank]*p["singular"][:rank])@p["u"][:,:rank].T
                native_delta={}
                for prefix in config["native_prefixes"]:
                    selected=p["order"][:prefix]
                    native_delta[prefix]=(final@p["raw_context_rows"][selected].T)@block.mlp.Down.weight[:,selected].T
                for scale in config["edit_scales"]:
                    edited=final+scale*p["edit"]
                    exact_write=block.mlp.Down(block.mlp.Left(edited)*block.mlp.Right(edited))+block.mlp.Down_bias
                    exact_x=native_x.clone(); exact_x[:,-1]=pre[:,-1]+exact_write
                    exact_logits=run_suffix(model,exact_x,value_bus,x0,layer+1)
                    intervention=exact_kl(baseline,exact_logits)
                    key=f"address{index}/scale{scale:g}"; store[f"{key}/intervention"].extend(intervention.cpu().tolist())
                    variants={**{f"rank{rank}":exact_base+scale*delta for rank,delta in rank_delta.items()},**{f"top{prefix}":exact_base+scale*delta for prefix,delta in native_delta.items()}}
                    for name,replacement in variants.items():
                        changed=native_x.clone(); changed[:,-1]=pre[:,-1]+replacement
                        logits=run_suffix(model,changed,value_bus,x0,layer+1)
                        store[f"{key}/{name}"].extend(exact_kl(exact_logits,logits).cpu().tolist())
        corpus_rows={}
        for index in range(len(programs)):
            for scale in config["edit_scales"]:
                key=f"address{index}/scale{scale:g}"; intervention=np.asarray(store[f"{key}/intervention"]); denom=max(intervention.mean(),1e-30)
                row={"mean_intervention_kl":float(intervention.mean()),"rank_frontier":{},"native_frontier":{}}
                for rank in config["ranks"]:
                    error=np.asarray(store[f"{key}/rank{rank}"]); row["rank_frontier"][str(rank)]={"mean_prediction_kl":float(error.mean()),"relative_prediction_kl":float(error.mean()/denom)}
                for prefix in config["native_prefixes"]:
                    error=np.asarray(store[f"{key}/top{prefix}"]); row["native_frontier"][str(prefix)]={"mean_prediction_kl":float(error.mean()),"relative_prediction_kl":float(error.mean()/denom)}
                corpus_rows[key]=row
        result[corpus]=corpus_rows
    return result


def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source"]).read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    stability={}; substitutions={}; causal={}; constructed={}
    for layer in config["layers"]:
        print("layer",layer,"two-slice construction",flush=True)
        edits_a,programs_a,_,_,cov_a=construct(model,layer,config["discovery_a"],config)
        edits_b,programs_b,_,inverse_b,cov_b=construct(model,layer,config["discovery_b"],config)
        qa=torch.linalg.qr(edits_a,mode="reduced").Q; qb=torch.linalg.qr(edits_b,mode="reduced").Q
        canonical=torch.linalg.svdvals(qa.T@qb).cpu().tolist(); overlap=(edits_a.T@edits_b).abs().cpu().numpy(); permutation,mean_match=best_permutation(overlap)
        stability[str(layer)]={"canonical_cosines":canonical,"minimum_canonical_cosine":min(canonical),"best_permutation":permutation,"mean_matched_absolute_cosine":mean_match,"matched_absolute_cosines":[float(overlap[i,permutation[i]]) for i in range(4)],"slice_a_k90":[p["operator_k90"] for p in programs_a],"slice_b_k90":[p["operator_k90"] for p in programs_b],"slice_a_covariance":cov_a,"slice_b_covariance":cov_b}
        polarities={}
        for index,features in source["recurrent_features"][str(layer)].items():
            for feature in features:
                if feature.startswith("current/newline/"): polarities[index]=feature.rsplit("/",1)[1]
        print("layer",layer,"newline substitutions",flush=True)
        substitutions[str(layer)]=newline_substitution(model,tokenizer,layer,programs_b,config,polarities)
        print("layer",layer,"rank/native frontiers",flush=True)
        causal[str(layer)]=causal_frontiers(model,layer,programs_b,inverse_b,config)
        constructed[str(layer)]={"k90":[p["operator_k90"] for p in programs_b],"polarities":polarities}

    compact=[(layer,index,k) for layer,row in constructed.items() for index,k in enumerate(row["k90"]) if k<=2]
    def rel(layer,index,scale,family,key): return [causal[layer][corpus][f"address{index}/scale{scale:g}"][family][str(key)]["relative_prediction_kl"] for corpus in causal[layer]]
    def selected_rank(k):
        return next((rank for rank in config["ranks"] if rank >= k), config["ranks"][-1])
    validated=[]
    for layer,row in constructed.items():
        for index in row["polarities"]:
            if all(substitutions[layer][corpus][index]["predicted_direction_fraction"]>=.75 for corpus in substitutions[layer]): validated.append((layer,int(index)))
    selected_scale={}
    for index,k in enumerate(constructed["11"]["k90"]):
        selected_scale[str(index)]={str(scale):float(np.median(rel("11",index,scale,"rank_frontier",k))) for scale in config["edit_scales"]}
    hypotheses={
        "R1_layer11_address_subspace_is_stable":{"supported":stability["11"]["minimum_canonical_cosine"]>=.90},
        "R2_layer10_address_subspace_is_less_stable":{"supported":stability["10"]["minimum_canonical_cosine"]<=stability["11"]["minimum_canonical_cosine"]-.10},
        "R3_k90_rank_predicts_compact_edits":{"supported":all(max(rel(layer,index,.25,"rank_frontier",k))<=.10 for layer,index,k in compact)},
        "R4_rank4_predicts_all_layer11_edits":{"supported":all(max(rel("11",index,.25,"rank_frontier",4))<=.10 for index in range(4))},
        "R5_top256_native_channels_predict_all_layer11_edits":{"supported":all(max(rel("11",index,.25,"native_frontier",256))<=.10 for index in range(4))},
        "R6_observational_boundary_features_survive_token_edits":{"supported":len(validated)>=5,"validated":validated},
        "R7_rank_frontier_is_scale_stable":{"supported":all(selected_scale[str(i)]["1.0"]<=2*selected_scale[str(i)]["0.25"] for i in range(4)),"selected_rank_relative_errors":selected_scale},
        "R8_layer11_compactness_is_slice_stable":{"supported":all(k<=4 for k in stability["11"]["slice_a_k90"]+stability["11"]["slice_b_k90"])},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"stability":stability,"constructed":constructed,"newline_substitution":substitutions,"causal":causal},indent=2,allow_nan=False))

    fig,axes=plt.subplots(1,3,figsize=(15,4.5)); labels=[f"L{l}A{i}" for l in ("10","11") for i in range(4)]; x=np.arange(8)
    axes[0].bar(np.arange(4)-.18,stability["10"]["canonical_cosines"],.36,label="L10"); axes[0].bar(np.arange(4)+.18,stability["11"]["canonical_cosines"],.36,label="L11"); axes[0].set_title("Address-subspace canonical cosines"); axes[0].legend()
    axes[1].bar(x,[np.median(rel(l,i,.25,"rank_frontier",selected_rank(constructed[l]["k90"][i]))) for l in ("10","11") for i in range(4)]); axes[1].axhline(.1,color="black",ls="--",lw=1); axes[1].set_title("Selected-rank relative KL")
    fractions=[]
    for l in ("10","11"):
        for i in range(4):
            key=str(i); vals=[substitutions[l][c][key]["predicted_direction_fraction"] for c in substitutions[l] if key in substitutions[l][c]]; fractions.append(np.mean(vals) if vals else np.nan)
    axes[2].bar(x, fractions); axes[2].axhline(.75,color="black",ls="--",lw=1); axes[2].set_title("Newline-edit direction fidelity")
    for ax in axes[1:]: ax.set_xticks(x,labels,rotation=45); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Late-program rank and semantic red team","","This follow-up tests whether the compact late programs are stable across covariance samples, whether the appropriate rank rather than rank one predicts behavior, whether larger native prefixes close the gap, and whether observational newline features survive literal input edits.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Late red team](bilin12_late_program_rank_semantic_redteam.png)","",f"Layer-10 canonical cosines: `{[round(x,3) for x in stability['10']['canonical_cosines']]}`; Layer-11: `{[round(x,3) for x in stability['11']['canonical_cosines']]}`.","", "| Program | K90 | Reported rank | Selected-rank rel. KL @0.25 | Rank4 rel. KL | Top256 rel. KL | Newline-edit fidelity |", "|---|---:|---:|---:|---:|---:|---:|"]
    for l in ("10","11"):
        for i,k in enumerate(constructed[l]["k90"]):
            reported_rank=selected_rank(k); rk=np.median(rel(l,i,.25,"rank_frontier",reported_rank)); r4=np.median(rel(l,i,.25,"rank_frontier",4)); top=np.median(rel(l,i,.25,"native_frontier",256)); key=str(i); vals=[substitutions[l][c][key]["predicted_direction_fraction"] for c in substitutions[l] if key in substitutions[l][c]]
            lines.append(f"| L{l}A{i} | {k} | {reported_rank} | {rk:.3f} | {r4:.3f} | {top:.3f} | {np.mean(vals):.3f} |" if vals else f"| L{l}A{i} | {k} | {reported_rank} | {rk:.3f} | {r4:.3f} | {top:.3f} | n/a |")
    lines += ["","Canonical subspace stability tests whether the explanatory address family is reproducible even if individual axes rotate. Rank-K90 tests the operator in its intrinsic basis; native prefixes test the model's own neuron coordinates. Literal token replacement is stronger than observational AUROC, although it remains a nonminimal intervention because changing a token also changes its contextual state."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
