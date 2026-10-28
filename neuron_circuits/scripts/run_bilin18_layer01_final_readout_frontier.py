#!/usr/bin/env python3
"""Native-versus-rotated rank frontier for conditional final-query readouts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_native_rotation_tournament import make_rotation,rotate,selected_native,selected_rotated
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT,fisher_fidelity


CONFIG=Path("configs/bilin18_layer01_final_readout_frontier.json")
OUTPUT=TENSOR_ROOT/"layer01_final_readout_frontier.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_READOUT_FRONTIER.md"
FIGURE=TENSOR_ROOT/"layer01_final_readout_frontier.png"
METHODS=("native_activation","native_local","rotated_local")


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def first_threshold(curve,ranks,threshold):
    for rank,value in zip(ranks,curve):
        if value>=threshold: return rank
    return None


@torch.inference_mode()
def evaluate(model,layer,base,full,deltas,position):
    block=model.blocks[layer]; names=list(deltas); outputs=[base["output"]]; x0s=[base["x0"]]; buses=[base["value_bus"]]
    for delta in list(deltas.values())+[full]:
        output=base["output"].clone(); output[:,position]+=F.linear(delta,block.mlp.Down.weight.float()).to(output.dtype); outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits=suffix_from_layer(model,layer,torch.cat(x0s),torch.cat(buses),torch.cat(outputs),position); base_logits=logits[0]; target=logits[-1]-base_logits
    return {name:fisher_fidelity(base_logits,target,logits[index+1]-base_logits)[0] for index,name in enumerate(names)}


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); bilinear=json.loads(Path(config["source_config"]).read_text()); final_config=json.loads(Path(bilinear["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source); new_config=json.loads(Path(config["new_pair_config"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; results={}; maximum_rotation_error=0.
    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); permutations=make_rotation(decoder.shape[1],config["rotation_rounds"],config["rotation_seed"]+layer,"cuda"); rotated_decoder=rotate(decoder,permutations); records=[]
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; base_state=F.rms_norm(base["pre"][0,final_position].float(),(model.config.n_embd,)); counter_state=F.rms_norm(counter["pre"][0,final_position].float(),(model.config.n_embd,)); full=block.mlp.Left(counter_state)*block.mlp.Right(counter_state)-block.mlp.Left(base_state)*block.mlp.Right(base_state); target_write=F.linear(full,decoder); native_score=(full*(target_write@decoder)).abs(); rotated_full=rotate(full,permutations); rotated_score=(rotated_full*(target_write@rotated_decoder)).abs(); restored=selected_rotated(full,torch.arange(len(full),device="cuda"),permutations); maximum_rotation_error=max(maximum_rotation_error,float((restored-full).norm()/full.norm().clamp_min(1e-30))); deltas={}
                        for rank in config["ranks"]:
                            deltas[f"native_activation_{rank}"]=selected_native(full,full.abs().topk(rank).indices); deltas[f"native_local_{rank}"]=selected_native(full,native_score.topk(rank).indices); deltas[f"rotated_local_{rank}"]=selected_rotated(full,rotated_score.topk(rank).indices,permutations)
                        effects=evaluate(model,layer,base,full,deltas,final_position); records.append({"corpus":corpus,"family":family_name,"pair":f"{source_token}->{target_token}","context":context_index,"fidelity":effects})
                    print("readout frontier",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]; methods={method:[median([row["fidelity"][f"{method}_{rank}"] for row in selected]) for rank in config["ranks"]] for method in METHODS}; summaries[corpus]={"curves":methods,"s50":{method:first_threshold(curve,config["ranks"],.5) for method,curve in methods.items()},"s90":{method:first_threshold(curve,config["ranks"],.9) for method,curve in methods.items()}}
        results[str(layer)]={"summaries":summaries,"records":records}
    def row(layer,corpus): return results[str(layer)]["summaries"][corpus]
    rank_index=config["ranks"].index(256); advantages={(l,c):row(l,c)["curves"]["native_local"][rank_index]-row(l,c)["curves"]["rotated_local"][rank_index] for l in config["layers"] for c in fixed["evaluation"]}
    hypotheses={
        "C0_native_local_reaches_half_by_256":{"supported":all(row(l,c)["s50"]["native_local"] is not None and row(l,c)["s50"]["native_local"]<=256 for l in config["layers"] for c in fixed["evaluation"])},
        "C1_high_fidelity_is_not_sparse":{"supported":all(row(l,c)["s90"]["native_local"] is not None and row(l,c)["s90"]["native_local"]>=1024 for l in config["layers"] for c in fixed["evaluation"])},
        "C2_native_basis_beats_rotation":{"supported":all(advantages[(l,c)]>=.05 for l in config["layers"] for c in fixed["evaluation"])},
        "C3_activation_ranking_nearly_matches_local":{"supported":all(row(l,c)["curves"]["native_activation"][rank_index]>=row(l,c)["curves"]["native_local"][rank_index]-.1 for l in config["layers"] for c in fixed["evaluation"])},
        "C4_full_frontier_reconstructs":{"supported":all(row(l,c)["curves"][method][-1]>=.9 for l in config["layers"] for c in fixed["evaluation"] for method in METHODS)},
        "C5_native_advantage_transfers_ood":{"supported":all(abs(advantages[(l,"fineweb")]-advantages[(l,"tinystories")])<=.1 for l in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_rotation_error":maximum_rotation_error,"layers":results},indent=2)); lines=["# Native-versus-rotated frontier for conditional final-query readouts","","For each prospective edit, channels are selected anew from the realized final-position bilinear product change. Native activation magnitude and exact local-write contribution are compared with local contribution after an exact dense orthogonal product-coordinate rotation. Every selected delta is propagated through the complete suffix.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Native activation S50/S90 | Native local S50/S90 | Rotated local S50/S90 | Rank-256 native/rotated advantage |","|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); fmt=lambda method:f"{item['s50'][method]}/{item['s90'][method]}"; lines.append(f"| {layer} | {corpus} | {fmt('native_activation')} | {fmt('native_local')} | {fmt('rotated_local')} | {item['curves']['native_local'][rank_index]:.3f}/{item['curves']['rotated_local'][rank_index]:.3f} ({advantages[(layer,corpus)]:+.3f}) |")
    lines += ["",f"Maximum full-coordinate rotation reconstruction error: **{maximum_rotation_error:.2e}**.","","Native advantage here means the architectural product coordinates are unusually useful for sparse **instance-conditioned** execution. It does not imply that the same neurons recur across examples or possess a fixed semantic label."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); colors={"native_activation":"tab:blue","native_local":"tab:green","rotated_local":"tab:red"}
    for axis,layer in zip(axes,config["layers"]):
        for corpus,style in zip(fixed["evaluation"],("-","--")):
            for method in METHODS: axis.plot(config["ranks"],row(layer,corpus)["curves"][method],style,marker="o",color=colors[method],label=f"{corpus}/{method}")
        axis.set_xscale("log",base=2); axis.set_ylim(-.05,1.05); axis.set_title(f"Layer {layer}"); axis.set_xlabel("selected coordinates"); axis.set_ylabel("downstream fidelity"); axis.grid(alpha=.25); axis.legend(fontsize=6,ncol=2)
    fig.savefig(FIGURE,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
