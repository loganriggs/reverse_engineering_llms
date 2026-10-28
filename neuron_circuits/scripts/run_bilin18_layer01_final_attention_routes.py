#!/usr/bin/env python3
"""Exact final-query QK1/QK2/value/head/source routes for Layers 0 and 1."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_semantic_source_factorial import attribution, fidelity, shapley
from run_bilin18_layer8_semantic_attention_atlas import forward_layers, three_factor_shapley
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_final_attention_routes.json")
OUTPUT=TENSOR_ROOT/"layer01_final_attention_routes.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_ATTENTION_ROUTES.md"
FIGURE=TENSOR_ROOT/"layer01_final_attention_routes.png"
FACTORS=("qk1","qk2","value")


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def coalition_keys(count): return list(itertools.product((0,1),repeat=count))


def rankdata(values: torch.Tensor) -> torch.Tensor:
    order=torch.argsort(values)
    ranks=torch.empty_like(order,dtype=torch.float32)
    ranks[order]=torch.arange(len(values),device=values.device,dtype=torch.float32)
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    a=rankdata(left.float()); b=rankdata(right.float()); a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))


def static_head_capacity(block) -> torch.Tensor:
    """A scale-canonicalized, state-free QK1×QK2×OV capacity score.

    Q/K module Frobenius scales are removed because each projected head is
    RMS-normalized.  OV is composed before taking its Frobenius norm, making
    that factor invariant to reciprocal linear value/output rescaling.
    """
    heads,head_dim=block.attn.n_head,block.attn.head_dim; scores=[]
    for head in range(heads):
        sl=slice(head*head_dim,(head+1)*head_dim)
        q=block.attn.c_q.weight.float()[sl]; k=block.attn.c_k.weight.float()[sl]
        q2=block.attn.c_q2.weight.float()[sl]; k2=block.attn.c_k2.weight.float()[sl]
        v=block.attn.c_v.weight.float()[sl]; o=block.attn.c_proj.weight.float()[:,sl]
        b1=(q/q.norm().clamp_min(1e-30)).T@(k/k.norm().clamp_min(1e-30))
        b2=(q2/q2.norm().clamp_min(1e-30)).T@(k2/k2.norm().clamp_min(1e-30))
        ov=o@v
        scores.append(b1.norm()*b2.norm()*ov.norm())
    return torch.stack(scores)


def project_head_edges(block, values: torch.Tensor) -> torch.Tensor:
    """Map [head, source, d_head] edge values to [head, source, d_model]."""
    rows=[]
    for head in range(block.attn.n_head):
        sl=slice(head*block.attn.head_dim,(head+1)*block.attn.head_dim)
        rows.append(values[head]@block.attn.c_proj.weight.float()[:,sl].T)
    return torch.stack(rows)


def attention_factorization(block, base, counter, position):
    d0,d1=base["details"],counter["details"]; sources=slice(0,position+1)
    a0=d0["score1"][0,:,position:position+1,sources].float(); a1=d1["score1"][0,:,position:position+1,sources].float()
    b0=d0["score2"][0,:,position:position+1,sources].float(); b1=d1["score2"][0,:,position:position+1,sources].float()
    v0=d0["mixed_value"][0,sources].float().permute(1,0,2); v1=d1["mixed_value"][0,sources].float().permute(1,0,2)
    pieces=three_factor_shapley(a0,a1,b0,b1,v0,v1)
    factor_edges={name:project_head_edges(block,value[:,0]) for name,value in pieces.items()}
    full_native=counter["attention_write"][0,position].float()-base["attention_write"][0,position].float()
    recomposed=sum(value.sum((0,1)) for value in factor_edges.values())
    raw=((a1*b1)[...,None]*v1[:,None]-(a0*b0)[...,None]*v0[:,None])[:,0]
    head_edges=project_head_edges(block,raw)
    head_writes=head_edges.sum(1)
    algebraic_target=head_writes.sum(0)
    algebraic_error=float((recomposed-algebraic_target).norm()/algebraic_target.norm().clamp_min(1e-30))
    native_order_error=float((algebraic_target-full_native).norm()/full_native.norm().clamp_min(1e-30))
    return factor_edges,head_edges,head_writes,full_native,algebraic_error,native_order_error


def selected_head_fidelity(target,head_writes,indices):
    return fidelity(target,head_writes[indices].sum(0))


def finite_mlp_factorial(block,base_pre,components,bank):
    dimension=base_pre.shape[-1]; base_state=F.rms_norm(base_pre,(dimension,)); base_product=block.mlp.Left(base_state)*block.mlp.Right(base_state); decoder=block.mlp.Down.weight.float()
    outputs={kind:{} for kind in ("full","family")}
    for key in coalition_keys(len(components)):
        pre=base_pre+sum((component for bit,component in zip(key,components.values()) if bit),torch.zeros_like(base_pre))
        state=F.rms_norm(pre,(dimension,)); delta=block.mlp.Left(state)*block.mlp.Right(state)-base_product
        outputs["full"][key]=F.linear(delta,decoder)
        selected=torch.zeros_like(delta); selected[...,bank]=delta[...,bank]
        outputs["family"][key]=F.linear(selected,decoder)
    result={}; maximum_error=0.
    for kind,worlds in outputs.items():
        target=worlds[(1,)*len(components)]-worlds[(0,)*len(components)]; phis=shapley(worlds); error=float((sum(phis)-target).norm()/target.norm().clamp_min(1e-30)); maximum_error=max(maximum_error,error)
        result[kind]={name:{"projection":attribution(target,phis[index])[0],"energy":attribution(target,phis[index])[1],"singleton_fidelity":fidelity(target,worlds[tuple(int(i==index) for i in range(len(components)))]-worlds[(0,)*len(components)])} for index,name in enumerate(components)}
    return result,maximum_error


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); parent=json.loads(Path(config["source_config"]).read_text()); source_validation=json.loads(Path(parent["source_config"]).read_text())
    # Recover the fixed dataset spec through the already validated config chain.
    validation_config=json.loads(Path(source_validation["source_config"]).read_text()); core_config=json.loads(Path(validation_config["source_config"]).read_text()); bank_config=json.loads(Path(core_config["source_config"]).read_text()); frontier_config=json.loads(Path(bank_config["source_config"]).read_text()); red_config=json.loads(Path(frontier_config["source_config"]).read_text()); tournament_config=json.loads(Path(red_config["source_config"]).read_text())
    from run_bilin18_layer3_frozen_channel_transfer import recover_configs
    fixed,_=recover_configs(tournament_config)
    bank_results=json.loads(Path(config["source_banks"]).read_text()); new_config=json.loads(Path(config["new_pair_config"]).read_text())
    edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; results={}; maximum_algebraic_error=0.; maximum_native_order_error=0.; maximum_shapley_error=0.
    for layer in config["layers"]:
        block=model.blocks[layer]; capacity=static_head_capacity(block).cuda(); banks={family:torch.tensor(row["final"],device="cuda") for family,row in bank_results["layers"][str(layer)]["bank_indices"].items()}; records=[]
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]
                        factor_edges,head_edges,head_writes,attention_target,algebraic_error,native_order_error=attention_factorization(block,base,counter,final_position); maximum_algebraic_error=max(maximum_algebraic_error,algebraic_error); maximum_native_order_error=max(maximum_native_order_error,native_order_error)
                        components={}
                        if layer>0: components["carried"]=counter["mixed"][0,final_position].float()-base["mixed"][0,final_position].float()
                        components.update({name:value.sum((0,1)) for name,value in factor_edges.items()})
                        factorial,factorial_error=finite_mlp_factorial(block,base["pre"][0,final_position].float(),components,banks[family]); maximum_shapley_error=max(maximum_shapley_error,factorial_error)
                        denominator=attention_target.square().sum().clamp_min(1e-30); head_projection=(head_writes@attention_target)/denominator; head_norm=head_writes.norm(dim=1)
                        count=min(config["top_heads"],len(head_norm)); selections={"static":torch.topk(capacity,count).indices,"state_norm":torch.topk(head_norm,count).indices,"target":torch.topk(head_projection.abs(),count).indices}
                        head_fidelities={name:selected_head_fidelity(attention_target,head_writes,index) for name,index in selections.items()}
                        edge_projection=(head_edges@attention_target)/denominator
                        source_projection={"before_edit":float(edge_projection[:,:edit_position].sum()),"edit":float(edge_projection[:,edit_position].sum()),"after_edit":float(edge_projection[:,edit_position+1:].sum())}
                        records.append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","context":context_index,"source_projection":source_projection,"factorial":factorial,"head_fidelity":head_fidelities,"static_spearman":spearman(capacity,head_projection.abs()),"top_heads":{name:index.cpu().tolist() for name,index in selections.items()}})
                    print("final attention",layer,corpus,family,source_token,target_token,flush=True)
        summaries={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]
            summaries[corpus]={
                "source_projection":{name:median([row["source_projection"][name] for row in selected]) for name in ("before_edit","edit","after_edit")},
                "full_factor_projection":{name:median([row["factorial"]["full"][name]["projection"] for row in selected]) for name in selected[0]["factorial"]["full"]},
                "full_factor_singleton":{name:median([row["factorial"]["full"][name]["singleton_fidelity"] for row in selected]) for name in selected[0]["factorial"]["full"]},
                "family_factor_projection":{name:median([row["factorial"]["family"][name]["projection"] for row in selected]) for name in selected[0]["factorial"]["family"]},
                "head_fidelity":{name:median([row["head_fidelity"][name] for row in selected]) for name in ("static","state_norm","target")},
                "static_spearman":median([row["static_spearman"] for row in selected]),
            }
        results[str(layer)]={"static_capacity":capacity.cpu().tolist(),"summaries":summaries,"records":records}
    def summary(layer,corpus): return results[str(layer)]["summaries"][corpus]
    corpora=list(fixed["evaluation"])
    hypotheses={
        "Q0_exact_factorization":{"supported":max(maximum_algebraic_error,maximum_native_order_error,maximum_shapley_error)<=2e-5,"maximum_relative_error":max(maximum_algebraic_error,maximum_native_order_error,maximum_shapley_error)},
        "Q1_layer0_is_single_source_transport":{"supported":all(summary(0,c)["source_projection"]["edit"]>=.9 for c in corpora)},
        "Q2_layer1_recruits_earlier_context":{"supported":all(abs(summary(1,c)["source_projection"]["before_edit"])>=.1 for c in corpora)},
        "Q3_layer0_value_path_dominates":{"supported":all(summary(0,c)["full_factor_projection"]["value"]>=.5 for c in corpora)},
        "Q4_layer1_routing_is_material":{"supported":all(summary(1,c)["full_factor_projection"]["qk1"]+summary(1,c)["full_factor_projection"]["qk2"]>=.25 for c in corpora)},
        "Q5_three_state_selected_heads_are_sufficient":{"supported":all(summary(l,c)["head_fidelity"]["target"]>=.6 for l in config["layers"] for c in corpora)},
        "Q6_state_selection_beats_static_capacity":{"supported":all(summary(l,c)["head_fidelity"]["target"]-summary(l,c)["head_fidelity"]["static"]>=.1 for l in config["layers"] for c in corpora)},
        "Q7_static_capacity_is_not_a_usage_map":{"supported":all(summary(l,c)["static_spearman"]<.5 for l in config["layers"] for c in corpora)},
        "Q8_no_single_QKOV_factor_is_sufficient":{"supported":all(max(summary(l,c)["full_factor_singleton"].get(name,-1e9) for name in FACTORS)<.9 for l in config["layers"] for c in corpora)},
        "Q9_factor_ordering_transfers_ood":{"supported":all(sorted(FACTORS,key=lambda name:summary(l,"fineweb")["full_factor_projection"][name])==sorted(FACTORS,key=lambda name:summary(l,"tinystories")["full_factor_projection"][name]) for l in config["layers"])},
    }
    diagnostics={"maximum_algebraic_factor_error":maximum_algebraic_error,"maximum_native_float32_order_error":maximum_native_order_error,"maximum_finite_shapley_error":maximum_shapley_error,"algebraic_identity_below_2e-5":maximum_algebraic_error<=2e-5 and maximum_shapley_error<=2e-5}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"diagnostics":diagnostics,"layers":results},indent=2))
    lines=["# Exact final-query attention routes in Layers 0 and 1","","For seventeen held-out lexical substitutions, the final-position attention change is decomposed exactly into native `(QK1)(QK2)V` source edges and heads. QK1, QK2, value, and (at Layer 1) carried-state boundary components are then passed through RMSNorm and the bilinear MLP in every finite coalition. Static head capacity uses only scale-canonicalized QK weight forms and composed OV weights; state and target rankings use the realized edit.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Source before/edit/after | Full-write factor projection | Head fidelity static/state/target | Static↔usage Spearman |","|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in corpora:
            row=summary(layer,corpus); src=row["source_projection"]; fac=row["full_factor_projection"]; heads=row["head_fidelity"]
            factor_text=" / ".join(f"{name}:{fac[name]:.3f}" for name in fac)
            lines.append(f"| {layer} | {corpus} | {src['before_edit']:.3f}/{src['edit']:.3f}/{src['after_edit']:.3f} | {factor_text} | {heads['static']:.3f}/{heads['state_norm']:.3f}/{heads['target']:.3f} | {row['static_spearman']:.3f} |")
    lines += ["",f"Maximum algebraic factor error: **{maximum_algebraic_error:.2e}**. Maximum finite-Shapley error: **{maximum_shapley_error:.2e}**. Maximum discrepancy to the checkpoint's native float32 contraction order: **{maximum_native_order_error:.2e}**.","", "The locked exactness gate retains the native-order discrepancy and therefore remains failed if that value exceeds `2e-5`; the separately reported algebraic identity is a post-run diagnostic, not a rewritten hypothesis.","","The static score is a state-free capacity measure, not a claim of unique canonical gauge. The state-selected score uses exact head deltas; the target-selected score is an oracle upper control. Source and factor projections are signed and can exceed one or be negative when routes cancel."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); cells=[(l,c) for l in config["layers"] for c in corpora]; x=np.arange(len(cells)); labels=[f"L{l} {c[:4]}" for l,c in cells]
    for offset,name in zip((-.25,0,.25),("static","state_norm","target")): axes[0].bar(x+offset,[summary(l,c)["head_fidelity"][name] for l,c in cells],.25,label=name)
    axes[0].set_xticks(x,labels,rotation=20); axes[0].set_ylabel("top-3 attention fidelity"); axes[0].legend(fontsize=8); axes[0].grid(axis="y",alpha=.25)
    width=.22
    for offset,name in zip((-.22,0,.22),FACTORS): axes[1].bar(x+offset,[summary(l,c)["full_factor_projection"][name] for l,c in cells],width,label=name)
    axes[1].set_xticks(x,labels,rotation=20); axes[1].set_ylabel("finite full-MLP projection"); axes[1].legend(fontsize=8); axes[1].grid(axis="y",alpha=.25)
    fig.savefig(FIGURE,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
