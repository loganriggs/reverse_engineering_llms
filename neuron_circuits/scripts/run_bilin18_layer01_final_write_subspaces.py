#!/usr/bin/env python3
"""Prospective causal test of compact final-position MLP write subspaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT,fisher_fidelity
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG=Path("configs/bilin18_layer01_final_write_subspaces.json")
OUTPUT=TENSOR_ROOT/"layer01_final_write_subspaces.json"
REPORT=TENSOR_ROOT/"LAYER01_FINAL_WRITE_SUBSPACES.md"
FIGURE=TENSOR_ROOT/"layer01_final_write_subspaces.png"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def fit_basis(rows,rank):
    matrix=torch.stack([row/row.norm().clamp_min(1e-30) for row in rows])
    return torch.linalg.svd(matrix,full_matrices=False).Vh[:min(rank,len(rows))].T.contiguous()


def project(value,basis): return (value@basis)@basis.T


def local_fidelity(target,prediction): return 1-float((prediction-target).square().sum()/target.square().sum().clamp_min(1e-30))


@torch.inference_mode()
def causal_variants(model,layer,base,deltas,position):
    names=[name for name in deltas if name!="full"]
    ordered=names+["full"]
    outputs=[base["output"]]; x0s=[base["x0"]]; buses=[base["value_bus"]]
    for name in ordered:
        delta=deltas[name]
        output=base["output"].clone(); output[:,position]+=delta.to(output.dtype); outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits=suffix_from_layer(model,layer,torch.cat(x0s),torch.cat(buses),torch.cat(outputs),position); base_logits=logits[0]; target=logits[-1]-base_logits
    return {name:{"downstream":fisher_fidelity(base_logits,target,logits[index+1]-base_logits)[0],"local":local_fidelity(deltas["full"],deltas[name])} for index,name in enumerate(names)}


@torch.inference_mode()
def main():
    config_path=Path(sys.argv[1]) if len(sys.argv)>1 else CONFIG; config=json.loads(config_path.read_text()); output_path=Path(config.get("output_path",OUTPUT)); report_path=Path(config.get("report_path",REPORT)); figure_path=Path(config.get("figure_path",FIGURE)); final_config=json.loads(Path(config["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family_config=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family_config["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,parent=recover_configs(source)
    saved=json.loads(Path(config["source_records"]).read_text()); bank_results=json.loads(Path(config["source_bank_results"]).read_text()); new_config=json.loads(Path(config["new_pair_config"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; discovery=[row for row in saved["semantic_records"] if row["corpus"]==config["discovery_corpus"] and row["context_index"] in config["discovery_context_indices"]]; family_names=list(parent["families"]); results={}
    def trace_saved(record,layer):
        spec=fixed["evaluation"][record["corpus"]]; rows=arrays[record["corpus"]][spec["start"]:spec["start"]+spec["rows"]]; raw=np.asarray(rows[record["row"],record["source_start"]:record["source_start"]+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=int(record["pair"].split("->")[1])
        return forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer],forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]
    for layer in config["layers"]:
        family_rows={name:[] for name in family_names}; all_rows=[]
        if "discovery_contexts_per_pair" in config:
            corpus=config["discovery_corpus"]; spec=fixed["evaluation"][corpus]; rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["discovery_seed"]+layer)
            for family_name,pairs in parent["families"].items():
                for source_token,target_token in pairs:
                    for row_index,start in select_windows(rows,source_token,edit_position,final_position,config["discovery_contexts_per_pair"],rng):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; delta=(counter["mlp_write"][0,final_position]-base["mlp_write"][0,final_position]).float(); family_rows[family_name].append(delta); all_rows.append(delta)
        else:
            for record in discovery:
                base,counter=trace_saved(record,layer); delta=(counter["mlp_write"][0,final_position]-base["mlp_write"][0,final_position]).float(); family_rows[record["family"]].append(delta); all_rows.append(delta)
        bases={rank:{"global":fit_basis(all_rows,rank),**{name:fit_basis(rows,rank) for name,rows in family_rows.items()}} for rank in config["ranks"]}
        generator=torch.Generator(device="cuda").manual_seed(config["random_seed"]+layer); random_full=torch.linalg.qr(torch.randn(model.config.n_embd,max(config["ranks"]),generator=generator,device="cuda"),mode="reduced").Q
        records=[]
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; target=(counter["mlp_write"][0,final_position]-base["mlp_write"][0,final_position]).float(); deltas={"full":target}; labels={}
                        for rank in config["ranks"]:
                            deltas[f"global_{rank}"]=project(target,bases[rank]["global"]); deltas[f"random_{rank}"]=project(target,random_full[:,:rank]); labels[rank]={}
                            for candidate in family_names: deltas[f"family_{candidate}_{rank}"]=project(target,bases[rank][candidate])
                        effects=causal_variants(model,layer,base,deltas,final_position); rank_rows={}
                        for rank in config["ranks"]:
                            correct=effects[f"family_{family_name}_{rank}"]; wrong=[effects[f"family_{candidate}_{rank}"]["downstream"] for candidate in family_names if candidate!=family_name]
                            rank_rows[str(rank)]={"correct":correct["downstream"],"correct_local":correct["local"],"wrong":median(wrong),"advantage":correct["downstream"]-median(wrong),"global":effects[f"global_{rank}"]["downstream"],"random":effects[f"random_{rank}"]["downstream"],"family_effects":{candidate:effects[f"family_{candidate}_{rank}"]["downstream"] for candidate in family_names}}
                        records.append({"corpus":corpus,"family":family_name,"pair":f"{source_token}->{target_token}","context":context_index,"ranks":rank_rows})
                    print("write subspace",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]; summaries[corpus]={str(rank):{key:median([row["ranks"][str(rank)][key] for row in selected]) for key in ("correct","correct_local","wrong","advantage","global","random")} for rank in config["ranks"]}
            for rank in config["ranks"]: summaries[corpus][str(rank)]["win_rate"]=float(np.mean([row["ranks"][str(rank)]["advantage"]>0 for row in selected]))
        results[str(layer)]={"discovery_counts":{name:len(rows) for name,rows in family_rows.items()},"summaries":summaries,"records":records,"native_bank_reference":bank_results["layers"][str(layer)]["summaries"]}
    maximum_rank=max(config["ranks"]); near_rank=sorted(config["ranks"])[-2]
    def row(layer,corpus,rank=maximum_rank): return results[str(layer)]["summaries"][corpus][str(rank)]
    common={
        "global":all(row(l,c)["global"]>=.5 for l in config["layers"] for c in fixed["evaluation"]),
        "specific":all(row(l,c)["advantage"]>=.05 for l in config["layers"] for c in fixed["evaluation"]),
        "family_global":all(row(l,c)["correct"]-row(l,c)["global"]>=.03 for l in config["layers"] for c in fixed["evaluation"]),
        "random":all(row(l,c)["correct"]-row(l,c)["random"]>=.2 for l in config["layers"] for c in fixed["evaluation"]),
        "ood":all(row(l,"tinystories")["correct"]>=.75*row(l,"fineweb")["correct"] for l in config["layers"]),
        "near":all(row(l,c,near_rank)["correct"]>=row(l,c)["correct"]-.1 for l in config["layers"] for c in fixed["evaluation"]),
    }
    if "discovery_contexts_per_pair" in config:
        hypotheses={
            "R0_rank24_global_transfers":{"supported":common["global"]},
            "R1_rank24_family_is_specific":{"supported":common["specific"]},
            "R2_rank24_beats_random":{"supported":common["random"]},
            "R3_more_data_rescues_rank6_failure":{"supported":all(row(l,c)["correct"]>=.3 for l in config["layers"] for c in fixed["evaluation"])},
            "R4_subspace_transfers_ood":{"supported":common["ood"]},
            "R5_rank16_is_near_rank24":{"supported":common["near"]},
        }
    else:
        hypotheses={
            "S0_compact_global_subspace_transfers":{"supported":common["global"]},
            "S1_family_subspace_is_specific":{"supported":common["specific"]},
            "S2_family_subspace_beats_global":{"supported":common["family_global"]},
            "S3_learned_subspace_beats_random":{"supported":common["random"]},
            "S4_subspace_transfers_ood":{"supported":common["ood"]},
            "S5_rank4_is_near_rank6":{"supported":common["near"]},
        }
    output_path.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2))
    discovery_description=(f"{config['discovery_contexts_per_pair']} newly selected FineWeb contexts per original lexical pair" if "discovery_contexts_per_pair" in config else "two saved FineWeb contexts per original lexical pair")
    lines=["# Final-query MLP write subspaces in Layers 0 and 1","",f"Unit-normalized final-position MLP write changes from the original nineteen substitutions and {discovery_description} define frozen family and global SVD bases. Ranks `{config['ranks']}` are projected into seventeen new substitutions and propagated through the complete suffix. Fixed random bases and wrong-family bases are causal controls.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["",f"| Layer | Corpus | Rank-{maximum_rank} correct / wrong / advantage (win) | Global | Random | Local correct | Native-256 reference |","|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); native=bank_results["layers"][str(layer)]["summaries"][corpus]["correct"]; lines.append(f"| {layer} | {corpus} | {item['correct']:.3f} / {item['wrong']:.3f} / {item['advantage']:.3f} ({item['win_rate']:.2f}) | {item['global']:.3f} | {item['random']:.3f} | {item['correct_local']:.3f} | {native:.3f} |")
    lines += ["",f"| Layer | Corpus | Family rank frontier {' / '.join(map(str,config['ranks']))} | Global frontier |","|---:|---|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            fam=" / ".join(f"{row(layer,corpus,rank)['correct']:.3f}" for rank in config["ranks"]); glob=" / ".join(f"{row(layer,corpus,rank)['global']:.3f}" for rank in config["ranks"]); lines.append(f"| {layer} | {corpus} | {fam} | {glob} |")
    lines += ["","A compact global subspace would support a reusable query-position effect interface even when native family banks fail. A family-over-global advantage is the stronger semantic claim; without it, the subspace is a generic edit/readout manifold rather than a named concept module."]
    report_path.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); cells=[(l,c) for l in config["layers"] for c in fixed["evaluation"]]; labels=[f"L{l} {c[:4]}" for l,c in cells]
    for axis,key,title in zip(axes,("correct","global"),("family","global")):
        for l,c in cells: axis.plot(config["ranks"],[row(l,c,r)[key] for r in config["ranks"]],marker="o",label=f"L{l} {c[:4]}")
        axis.set_title(title); axis.set_xlabel("write-subspace rank"); axis.set_ylabel("downstream fidelity"); axis.grid(alpha=.25); axis.legend(fontsize=7)
    fig.savefig(figure_path,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(report_path)


if __name__=="__main__": main()
