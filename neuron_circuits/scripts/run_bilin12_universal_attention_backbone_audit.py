#!/usr/bin/env python3
"""Search one attention-layer backbone jointly across corpora and edit distances."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_layer_subset_search import prepare, score
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_universal_attention_backbone_audit.json")
OUTPUT=TENSOR_ROOT/"bilin12_universal_attention_backbone_audit.json"
REPORT=TENSOR_ROOT/"BILIN12_UNIVERSAL_ATTENTION_BACKBONE_AUDIT.md"
FIGURE=TENSOR_ROOT/"bilin12_universal_attention_backbone_audit.png"


def global_score(model,caches,selected):
    rows={name:score(model,cache,selected) for name,cache in caches.items()}
    return float(np.mean([row["fisher_fidelity"] for row in rows.values()])),rows


@torch.inference_mode()
def prepare_light(model,tokenizer,rows,position,batch_size,sequence_length):
    """Keep only the native attention writes consumed by run_intervention."""
    full=prepare(model,tokenizer,rows,position,batch_size,sequence_length);light=[]
    for base_ids,base_rows,changed_rows,base_logits,changed_logits in full:
        base_light=[{"details":{"write":row["details"]["write"]}} for row in base_rows]
        changed_light=[{"details":{"write":row["details"]["write"]}} for row in changed_rows]
        light.append((base_ids,base_light,changed_light,base_logits,changed_logits))
    return light


def forward_search(model,caches):
    selected=[];sets={0:[]};scores={0:global_score(model,caches,[])[0]}
    for size in range(1,13):
        candidates=[]
        for layer in range(12):
            if layer in selected:continue
            value,_=global_score(model,caches,selected+[layer]);candidates.append((value,-layer,layer))
        value,_,layer=max(candidates);selected=selected+[layer];sets[size]=selected.copy();scores[size]=value
    return sets,scores


def backward_search(model,caches):
    selected=list(range(12));sets={12:selected.copy()};scores={12:global_score(model,caches,selected)[0]}
    for size in range(11,0,-1):
        candidates=[]
        for layer in selected:
            candidate=[item for item in selected if item!=layer];value,_=global_score(model,caches,candidate);candidates.append((value,-layer,layer,candidate))
        value,_,_,selected=max(candidates);sets[size]=selected.copy();scores[size]=value
    return sets,scores


def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");selection_caches={}
    for corpus,spec in config["evaluation"].items():
        rows=load_rows(spec);selection=rows[:config["selection_rows"]];validation=rows[config["selection_rows"]:config["selection_rows"]+config["validation_rows"]]
        for position in config["edit_positions"]:
            key=f"{corpus}/{position}";print("preparing selection",key,flush=True);selection_caches[key]=prepare_light(model,tokenizer,selection,position,config["batch_size"],config["sequence_length"])
    forward_sets,forward_selection=forward_search(model,selection_caches);backward_sets,backward_selection=backward_search(model,selection_caches);methods={"forward":forward_sets,"backward":backward_sets}
    del selection_caches;torch.cuda.empty_cache()
    validation={method:{str(k):{} for k in config["budgets"]} for method in methods};validation["fixed"]={name:{} for name in config["fixed_candidates"]}
    for corpus,spec in config["evaluation"].items():
        rows=load_rows(spec);validation_rows=rows[config["selection_rows"]:config["selection_rows"]+config["validation_rows"]]
        for position in config["edit_positions"]:
            key=f"{corpus}/{position}";print("validating",key,flush=True);cache=prepare_light(model,tokenizer,validation_rows,position,config["batch_size"],config["sequence_length"])
            for method,sets in methods.items():
                for k in config["budgets"]:validation[method][str(k)][key]=score(model,cache,sets[k])
            for name,layers in config["fixed_candidates"].items():validation["fixed"][name][key]=score(model,cache,layers)
            del cache;torch.cuda.empty_cache()
    better={str(k):max(("forward","backward"),key=lambda method:np.mean([row["fisher_fidelity"] for row in validation[method][str(k)].values()])) for k in config["budgets"]}
    chosen=lambda k:methods[better[str(k)]][k]
    values=lambda k:[row["fisher_fidelity"] for row in validation[better[str(k)]][str(k)].values()]
    fixed4568=[row["fisher_fidelity"] for row in validation["fixed"]["middle_core_4568"].values()]
    means_by_corpus={corpus:float(np.mean([validation[better["4"]]["4"][f"{corpus}/{p}"]["fisher_fidelity"] for p in config["edit_positions"]])) for corpus in config["evaluation"]}
    hypotheses={
      "U1_universal_four_layer_backbone_is_useful":{"supported":min(values(4))>=.65},
      "U2_universal_six_layer_backbone_is_nearly_sufficient":{"supported":min(values(6))>=.85},
      "U3_layers_five_and_six_are_in_the_global_core":{"supported":all({5,6}.issubset(set(sets[4])) for sets in (forward_sets,backward_sets))},
      "U4_global_search_prefers_the_observed_middle_core":{"supported":len(set(chosen(4))&{4,5,6,8})>=3},
      "U5_fixed_4568_core_generalizes":{"supported":min(fixed4568)>=.65},
      "U6_global_frontier_is_monotone":{"supported":all(b>=a-1e-6 for a,b in zip([min(values(k)) for k in config["budgets"]][:-1],[min(values(k)) for k in config["budgets"]][1:]))},
      "U7_one_global_set_does_not_solve_everything":{"supported":min(values(8))<.90},
      "U8_backbone_transfer_is_not_fineweb_specific":{"supported":all(abs(means_by_corpus[corpus]-means_by_corpus["fineweb"])<=.15 for corpus in ("wikitext","tinystories")),"means":means_by_corpus},
    }
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    payload={"config":config,"hypotheses":hypotheses,"forward_sets":{str(k):v for k,v in forward_sets.items()},"backward_sets":{str(k):v for k,v in backward_sets.items()},"selection_scores":{"forward":forward_selection,"backward":backward_selection},"better_method":better,"validation":validation};OUTPUT.write_text(json.dumps(payload,indent=2,allow_nan=False))
    budgets=config["budgets"];fig,axes=plt.subplots(1,3,figsize=(17,4.7));
    for method in ("forward","backward"):
        axes[0].plot(budgets,[np.mean([row["fisher_fidelity"] for row in validation[method][str(k)].values()]) for k in budgets],marker="o",label=f"{method} mean");axes[0].plot(budgets,[min(row["fisher_fidelity"] for row in validation[method][str(k)].values()) for k in budgets],marker="x",ls="--",label=f"{method} min")
    keys=list(validation[better["4"]]["4"]);x=np.arange(len(keys));axes[1].bar(x-.18,[validation[better["4"]]["4"][key]["fisher_fidelity"] for key in keys],.36,label=f"global {chosen(4)}");axes[1].bar(x+.18,[validation["fixed"]["middle_core_4568"][key]["fisher_fidelity"] for key in keys],.36,label="fixed 4568");axes[1].set_xticks(x,keys,rotation=45,ha="right",fontsize=7);axes[1].set_title("Universal K=4 held-out fidelity")
    frequency=np.zeros(12)
    for k in budgets:
        for layer in chosen(k):frequency[layer]+=1
    axes[2].bar(np.arange(12),frequency);axes[2].set_title("Layer recurrence across global budgets");axes[2].set_xlabel("layer")
    axes[0].axhline(.85,color="black",ls="--",lw=.8);axes[0].set_xlabel("budget");axes[0].set_title("Global backbone frontier");
    for axis in axes:
        axis.grid(alpha=.2)
        handles,labels=axis.get_legend_handles_labels()
        if handles:axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Universal attention-backbone audit","","One forward-greedy and one backward-elimination layer set are optimized jointly over all nine corpus-by-distance selection conditions, then evaluated on disjoint rows. Fixed candidate sets `{4,5,6,8}` and `{3,4,5,6}` test the recurrent middle-layer pattern without further search.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Universal backbone](bilin12_universal_attention_backbone_audit.png)","","| Budget | Better method/set | Mean fidelity | Minimum fidelity |","|---:|---|---:|---:|"]
    for k in budgets:
        rows=validation[better[str(k)]][str(k)];lines.append(f"| {k} | {better[str(k)]} / {chosen(k)} | {np.mean([r['fisher_fidelity'] for r in rows.values()]):.3f} | {min(r['fisher_fidelity'] for r in rows.values()):.3f} |")
    lines += ["","| Condition | Global K4 | Fixed 4568 | Fixed 3456 |","|---|---:|---:|---:|"]
    for key in validation[better["4"]]["4"]:lines.append(f"| {key} | {validation[better['4']]['4'][key]['fisher_fidelity']:.3f} | {validation['fixed']['middle_core_4568'][key]['fisher_fidelity']:.3f} | {validation['fixed']['early_middle_3456'][key]['fisher_fidelity']:.3f} |")
    lines += ["","A universal subset is a reusable intervention interface, not proof that those layers alone compute every route. Condition-specific layers can compensate for omitted writes, and replacement patches inject native changed-run messages whose internal content is analyzed separately by the QK/OV audit."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
