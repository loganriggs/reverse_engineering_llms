#!/usr/bin/env python3
"""Exact conditional bilinear readout at the Layer-0/1 final query position."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT,fisher_fidelity


CONFIG=Path("configs/bilin18_layer01_final_bilinear_readout.json")
OUTPUT=TENSOR_ROOT/"layer01_final_bilinear_readout.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_BILINEAR_READOUT.md"
FIGURE=TENSOR_ROOT/"layer01_final_bilinear_readout.png"
TERMS=("left_cross","right_cross","quadratic")


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def bilinear_terms(block,base_state,counter_state):
    left=block.mlp.Left(base_state); right=block.mlp.Right(base_state); dleft=block.mlp.Left(counter_state)-left; dright=block.mlp.Right(counter_state)-right
    return {"left_cross":dleft*right,"right_cross":left*dright,"quadratic":dleft*dright}


def projection(target,value): return float((target*value).sum()/target.square().sum().clamp_min(1e-30))


@torch.inference_mode()
def downstream_fidelities(model,layer,base,product_deltas,position):
    block=model.blocks[layer]; names=list(product_deltas); outputs=[base["output"]]; x0s=[base["x0"]]; buses=[base["value_bus"]]
    for delta in product_deltas.values():
        output=base["output"].clone(); output[:,position]+=F.linear(delta,block.mlp.Down.weight.float()).to(output.dtype); outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits=suffix_from_layer(model,layer,torch.cat(x0s),torch.cat(buses),torch.cat(outputs),position); base_logits=logits[0]; target=logits[names.index("full")+1]-base_logits
    return {name:fisher_fidelity(base_logits,target,logits[index+1]-base_logits)[0] for index,name in enumerate(names)}


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); final_config=json.loads(Path(config["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source); new_config=json.loads(Path(config["new_pair_config"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; results={}; maximum_error=0.
    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); decoder_norm=decoder.norm(dim=0); static_indices=decoder_norm.topk(config["rank"]).indices; records=[]
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; base_state=F.rms_norm(base["pre"][0,final_position].float(),(model.config.n_embd,)); counter_state=F.rms_norm(counter["pre"][0,final_position].float(),(model.config.n_embd,)); terms=bilinear_terms(block,base_state,counter_state); full=sum(terms.values()); direct=block.mlp.Left(counter_state)*block.mlp.Right(counter_state)-block.mlp.Left(base_state)*block.mlp.Right(base_state); error=float((full-direct).norm()/direct.norm().clamp_min(1e-30)); maximum_error=max(maximum_error,error); target_write=F.linear(full,decoder)
                        term_projection={name:projection(target_write,F.linear(value,decoder)) for name,value in terms.items()}
                        rankings={"static_norm":static_indices,"activation":full.abs().topk(config["rank"]).indices,"state_write":(full.abs()*decoder_norm).topk(config["rank"]).indices,"local_contribution":(full*(target_write@decoder)).abs().topk(config["rank"]).indices}
                        deltas={"full":full,**terms}
                        for name,indices in rankings.items():
                            selected=torch.zeros_like(full); selected[indices]=full[indices]; deltas[name]=selected
                        fidelities=downstream_fidelities(model,layer,base,deltas,final_position)
                        records.append({"corpus":corpus,"family":family_name,"pair":f"{source_token}->{target_token}","context":context_index,"term_projection":term_projection,"fidelity":fidelities,"recomposition_error":error})
                    print("bilinear readout",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]; summaries[corpus]={"term_projection":{name:median([row["term_projection"][name] for row in selected]) for name in TERMS},"fidelity":{name:median([row["fidelity"][name] for row in selected]) for name in ("left_cross","right_cross","quadratic","static_norm","activation","state_write","local_contribution")}}
        results[str(layer)]={"summaries":summaries,"records":records,"static_indices":static_indices.cpu().tolist()}
    def row(layer,corpus): return results[str(layer)]["summaries"][corpus]
    hypotheses={
        "B0_exact_bilinear_decomposition":{"supported":maximum_error<=2e-5,"maximum_relative_error":maximum_error},
        "B1_both_cross_terms_are_material":{"supported":all(all(abs(row(l,c)["term_projection"][name])>=.15 for name in ("left_cross","right_cross")) for l in config["layers"] for c in fixed["evaluation"])},
        "B2_quadratic_edit_term_is_minor":{"supported":all(abs(row(l,c)["term_projection"]["quadratic"])<=.25 for l in config["layers"] for c in fixed["evaluation"])},
        "B3_state_write_ranking_is_useful":{"supported":all(row(l,c)["fidelity"]["state_write"]>=.4 for l in config["layers"] for c in fixed["evaluation"])},
        "B4_local_contribution_ranking_recovers_the_readout":{"supported":all(row(l,c)["fidelity"]["local_contribution"]>=.7 for l in config["layers"] for c in fixed["evaluation"])},
        "B5_state_contraction_beats_static_norm":{"supported":all(row(l,c)["fidelity"]["state_write"]-row(l,c)["fidelity"]["static_norm"]>=.2 for l in config["layers"] for c in fixed["evaluation"])},
        "B6_conditional_program_transfers_ood":{"supported":all(row(l,"tinystories")["fidelity"]["state_write"]>=.8*row(l,"fineweb")["fidelity"]["state_write"] for l in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_recomposition_error":maximum_error,"layers":results},indent=2))
    lines=["# Exact conditional bilinear readout at the Layer-0/1 final query","","For every prospective lexical edit, the normalized pre-MLP state change is expanded exactly as `ΔL·R(base) + L(base)·ΔR + ΔL·ΔR`. Each term and four channel-selection rules are patched through the complete suffix. Static decoder norm is compared with activation magnitude, state-conditioned write magnitude, and the exact local contribution to the full MLP write.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Left/right/quadratic projection | Term downstream fidelity | Static / activation / state-write / local-contribution rank-256 |","|---:|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); p=item["term_projection"]; f=item["fidelity"]; lines.append(f"| {layer} | {corpus} | {p['left_cross']:.3f}/{p['right_cross']:.3f}/{p['quadratic']:.3f} | {f['left_cross']:.3f}/{f['right_cross']:.3f}/{f['quadratic']:.3f} | {f['static_norm']:.3f}/{f['activation']:.3f}/{f['state_write']:.3f}/{f['local_contribution']:.3f} |")
    lines += ["",f"Maximum product-rule recomposition error: **{maximum_error:.2e}**.","","`local_contribution` is target-aware only at the local MLP-write boundary: it uses the exact full write but no semantic label or downstream logits. `state_write` uses only the realized product change and decoder-column norms, so it is the stronger non-circular test of whether weight contraction identifies active native channels."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); cells=[(l,c) for l in config["layers"] for c in fixed["evaluation"]]; x=np.arange(len(cells)); labels=[f"L{l} {c[:4]}" for l,c in cells]
    for offset,name in zip((-.3,-.1,.1,.3),("static_norm","activation","state_write","local_contribution")): axes[0].bar(x+offset,[row(l,c)["fidelity"][name] for l,c in cells],.2,label=name)
    axes[0].set_xticks(x,labels); axes[0].set_ylabel("rank-256 downstream fidelity"); axes[0].legend(fontsize=8); axes[0].grid(axis="y",alpha=.25)
    for offset,name in zip((-.22,0,.22),TERMS): axes[1].bar(x+offset,[row(l,c)["term_projection"][name] for l,c in cells],.22,label=name)
    axes[1].set_xticks(x,labels); axes[1].set_ylabel("local-write projection"); axes[1].legend(fontsize=8); axes[1].grid(axis="y",alpha=.25); fig.savefig(FIGURE,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
