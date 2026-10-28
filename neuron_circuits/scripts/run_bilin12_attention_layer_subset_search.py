#!/usr/bin/env python3
"""Held-out forward/backward greedy search for compact attention-layer routes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin12_multilayer_attention_routing_audit import run_intervention
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_attention_layer_subset_search.json")
OUTPUT=TENSOR_ROOT/"bilin12_attention_layer_subset_search.json"
REPORT=TENSOR_ROOT/"BILIN12_ATTENTION_LAYER_SUBSET_SEARCH.md"
FIGURE=TENSOR_ROOT/"bilin12_attention_layer_subset_search.png"


@torch.inference_mode()
def prepare(model,tokenizer,rows,position,batch_size,sequence_length):
    newline=tokenizer.encode("\n",add_special_tokens=False)[0]; cache=[]
    for start in range(0,len(rows),batch_size):
        raw=np.asarray(rows[start:start+batch_size,:sequence_length],dtype=np.int64); base_ids=torch.as_tensor(raw,device="cuda"); keep=base_ids[:,position]!=newline
        if not bool(keep.any()): continue
        base_ids=base_ids[keep]; changed_ids=base_ids.clone();changed_ids[:,position]=newline
        base_rows,base_logits,_=trace(model,base_ids);changed_rows,changed_logits,_=trace(model,changed_ids)
        cache.append((base_ids,base_rows,changed_rows,base_logits,changed_logits))
    return cache


@torch.inference_mode()
def score(model,cache,selected):
    target_energy=prediction_energy=cross=error=forward_kl=target_error_kl=0.
    for base_ids,base_rows,changed_rows,base,changed in cache:
        hybrid=run_intervention(model,base_ids,base_rows,changed_rows,selected,"replacement","insertion")
        target=changed-base;prediction=hybrid-base;te,pe,co,er=fisher_terms(base,target,prediction)
        target_energy+=float(te.sum());prediction_energy+=float(pe.sum());cross+=float(co.sum());error+=float(er.sum());forward_kl+=float(exact_kl(base,hybrid).sum());target_error_kl+=float(exact_kl(changed,hybrid).sum())
    full_forward=sum(float(exact_kl(base,changed).sum()) for _,_,_,base,changed in cache);full_reverse=sum(float(exact_kl(changed,base).sum()) for _,_,_,base,changed in cache)
    return {"fisher_fidelity":1-error/max(target_energy,1e-30),"fisher_cosine":cross/max(np.sqrt(prediction_energy*target_energy),1e-30),"target_projection":cross/max(target_energy,1e-30),"forward_kl_ratio":forward_kl/max(full_forward,1e-30),"target_error_kl_ratio":target_error_kl/max(full_reverse,1e-30)}


def forward_search(model,cache):
    selected=[];sets={0:[]};scores={0:score(model,cache,[])}
    for size in range(1,13):
        candidates=[]
        for layer in range(12):
            if layer in selected: continue
            candidate=selected+[layer];row=score(model,cache,candidate);candidates.append((row["fisher_fidelity"],-layer,layer,row))
        _,_,chosen,row=max(candidates);selected=selected+[chosen];sets[size]=selected.copy();scores[size]=row
    return sets,scores


def backward_search(model,cache):
    selected=list(range(12));sets={12:selected.copy()};scores={12:score(model,cache,selected)}
    for size in range(11,0,-1):
        candidates=[]
        for layer in selected:
            candidate=[item for item in selected if item!=layer];row=score(model,cache,candidate);candidates.append((row["fisher_fidelity"],-layer,layer,row,candidate))
        _,_,removed,row,candidate=max(candidates);selected=candidate;sets[size]=selected.copy();scores[size]=row
    sets[0]=[];scores[0]=score(model,cache,[]);return sets,scores


def main():
    config=json.loads(CONFIG.read_text());ranking_config=json.loads(Path(config["ranking_source"]).read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");results={}
    for corpus,spec in config["evaluation"].items():
        rows=load_rows(spec);selection=rows[:config["selection_rows"]];validation=rows[config["selection_rows"]:config["selection_rows"]+config["validation_rows"]];results[corpus]={}
        for position in config["edit_positions"]:
            print("searching",corpus,position,flush=True);selection_cache=prepare(model,tokenizer,selection,position,config["batch_size"],config["sequence_length"]);validation_cache=prepare(model,tokenizer,validation,position,config["batch_size"],config["sequence_length"])
            forward_sets,forward_selection=forward_search(model,selection_cache);backward_sets,backward_selection=backward_search(model,selection_cache);frozen=ranking_config["rankings"][corpus][str(position)]
            methods={"forward":forward_sets,"backward":backward_sets,"frozen":{k:frozen[:k] for k in range(13)}};validation_rows={name:{str(k):score(model,validation_cache,sets[k]) for k in config["budgets"]} for name,sets in methods.items()}
            validation_rows["all"]={"12":score(model,validation_cache,list(range(12)))}
            results[corpus][str(position)]={"forward_order":forward_sets[12],"backward_removal_order":[next(layer for layer in range(12) if layer in backward_sets[size+1] and layer not in backward_sets[size]) for size in range(11,0,-1)],"sets":{name:{str(k):sets[k] for k in config["budgets"]} for name,sets in methods.items()},"selection_scores":{"forward":{str(k):forward_selection[k] for k in config["budgets"]},"backward":{str(k):backward_selection[k] for k in config["budgets"]}},"validation":validation_rows}
    def fidelity(c,p,m,k):return results[c][str(p)]["validation"][m][str(k)]["fisher_fidelity"]
    better={c:{str(p):{str(k):max(("forward","backward"),key=lambda m:fidelity(c,p,m,k)) for k in config["budgets"]} for p in config["edit_positions"]} for c in results}
    better_set=lambda c,p,k:results[c][str(p)]["sets"][better[c][str(p)][str(k)]][str(k)]
    conditions=[(c,p) for c in results for p in config["edit_positions"]]
    improvements=[max(fidelity(c,p,"forward",4),fidelity(c,p,"backward",4))-fidelity(c,p,"frozen",4) for c,p in conditions]
    hypotheses={
      "G1_forward_greedy_four_layers_are_sufficient":{"supported":all(fidelity(c,p,"forward",4)>=.75 for c,p in conditions)},
      "G2_some_four_layer_search_is_sufficient":{"supported":all(max(fidelity(c,p,"forward",4),fidelity(c,p,"backward",4))>=.75 for c,p in conditions)},
      "G3_six_layers_recover_nearly_all_behavior":{"supported":all(max(fidelity(c,p,"forward",6),fidelity(c,p,"backward",6))>=.90 for c,p in conditions)},
      "G4_search_improves_frozen_single_layer_ranking":{"supported":sum(value>=.05 for value in improvements)>=len(improvements)/2,"improvements":improvements},
      "G5_forward_and_backward_find_a_shared_core":{"supported":all(len(set(results[c][str(p)]["sets"]["forward"]["4"])&set(results[c][str(p)]["sets"]["backward"]["4"]))>=2 for c,p in conditions)},
      "G6_middle_layers_form_a_recurrent_core":{"supported":all(len(set(better_set(c,p,4))&set(range(4,8)))>=2 for c,p in conditions)},
      "G7_heldout_search_frontiers_are_monotone":{"supported":all(all(b>=a-1e-6 for a,b in zip([max(fidelity(c,p,"forward",k),fidelity(c,p,"backward",k)) for k in config["budgets"]][:-1],[max(fidelity(c,p,"forward",k),fidelity(c,p,"backward",k)) for k in config["budgets"]][1:])) for c,p in conditions)},
      "G8_no_single_universal_four_layer_subset":{"supported":len({tuple(sorted(better_set(c,p,4))) for c,p in conditions})>1},
    }
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"better_method":better,"results":results},indent=2,allow_nan=False))
    budgets=config["budgets"];fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for corpus in results:
        axes[0].plot(budgets,[np.mean([max(fidelity(corpus,p,"forward",k),fidelity(corpus,p,"backward",k)) for p in config["edit_positions"]]) for k in budgets],marker="o",label=corpus)
        axes[1].plot(budgets,[np.mean([fidelity(corpus,p,"frozen",k) for p in config["edit_positions"]]) for k in budgets],marker="o",label=corpus)
    frequency=np.zeros(12)
    for c,p in conditions:
        for layer in better_set(c,p,4):frequency[layer]+=1
    axes[2].bar(np.arange(12),frequency);axes[0].axhline(.9,color="black",ls="--",lw=.8);axes[0].set_title("Better greedy held-out frontier");axes[1].set_title("Frozen single-layer ranking frontier");axes[2].set_title("Layer frequency in better K=4 sets");axes[2].set_xlabel("layer")
    for axis in axes[:2]:axis.set_xlabel("layer budget");axis.grid(alpha=.2);axis.legend(fontsize=8)
    axes[2].grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Held-out search for compact attention-layer routes","","Forward selection and backward elimination optimize replacement-insertion Fisher fidelity on eight examples, then frozen subsets are evaluated on sixteen disjoint examples. A third frontier uses the earlier single-layer ranking without search on this panel.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Layer subset search](bilin12_attention_layer_subset_search.png)","","| Corpus | Position | Forward K4 set/fidelity | Backward K4 set/fidelity | Frozen K4 fidelity | Better K6 fidelity |","|---|---:|---:|---:|---:|---:|"]
    for c,p in conditions:
        row=results[c][str(p)];lines.append(f"| {c} | {p} | {row['sets']['forward']['4']} / {fidelity(c,p,'forward',4):.3f} | {row['sets']['backward']['4']} / {fidelity(c,p,'backward',4):.3f} | {fidelity(c,p,'frozen',4):.3f} | {max(fidelity(c,p,'forward',6),fidelity(c,p,'backward',6)):.3f} |")
    lines += ["","Search identifies a sufficient interface set, not unique ownership. Forward and backward procedures can find different sets because changing one attention write alters the states consumed by every later block. Held-out evaluation limits direct selection overfit but does not remove the off-manifold nature of partial write replacement."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
