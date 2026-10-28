#!/usr/bin/env python3
"""Red-team the low-energy, high-behavior final-destination attention message."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes,trace
from run_bilin12_attention_relay_topology_audit import group_worlds,vector_shapley
from run_bilin12_boundary_carrier_depth_audit import exact_kl,fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_attention_destination_metric_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin12_attention_destination_metric_redteam.json"
REPORT=TENSOR_ROOT/"BILIN12_ATTENTION_DESTINATION_METRIC_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin12_attention_destination_metric_redteam.png"
GROUPS=("edited_position","subsequent_context","final_position")
METRICS=("baseline_fisher","changed_fisher","removal_fisher","forward_kl","target_error_kl")


def scalar_shapley(world_values):
    result={};count=len(GROUPS)
    for index,name in enumerate(GROUPS):
        bit=1<<index;value=0.
        for mask in range(1<<count):
            if mask&bit:continue
            size=mask.bit_count();weight=math.factorial(size)*math.factorial(count-size-1)/math.factorial(count);value+=weight*(world_values[mask|bit]-world_values[mask])
        result[name]=float(value)
    return result


def empty_accumulator():
    return {"examples":0,"state_den":0.,"state_cross":{n:0. for n in GROUPS},"insert_base_den":0.,"insert_changed_den":0.,"remove_changed_den":0.,"insert_base_cross":{n:0. for n in GROUPS},"insert_changed_cross":{n:0. for n in GROUPS},"remove_changed_cross":{n:0. for n in GROUPS},"forward_world_kl":[0.]*8,"target_world_kl":[0.]*8,"write_error":0.,"write_relative_num":0.,"write_relative_den":0.,"logit_error":0.}


def update(acc,write_worlds,insert_logits,remove_logits,changed_write,independent_changed_write_logits):
    write_target=write_worlds[7]-write_worlds[0];write_den=write_target.square().sum().clamp_min(1e-30);acc["state_den"]+=float(write_den)
    for name,effect in vector_shapley(GROUPS,write_worlds).items():acc["state_cross"][name]+=float((effect*write_target).sum())
    insert_target=insert_logits[7]-insert_logits[0];remove_target=remove_logits[7]-remove_logits[0]
    for prefix,reference,target,worlds in (("insert_base",insert_logits[0],insert_target,insert_logits),("insert_changed",insert_logits[7],insert_target,insert_logits),("remove_changed",remove_logits[7],remove_target,remove_logits)):
        energy,_,_,_=fisher_terms(reference,target,target);acc[f"{prefix}_den"]+=float(energy.sum())
        for name,effect in vector_shapley(GROUPS,worlds).items():
            _,_,cross,_=fisher_terms(reference,target,effect);acc[f"{prefix}_cross"][name]+=float(cross.sum())
    for mask in range(8):acc["forward_world_kl"][mask]+=float(exact_kl(insert_logits[0],insert_logits[mask]).sum());acc["target_world_kl"][mask]+=float(exact_kl(insert_logits[7],insert_logits[mask]).sum())
    error=write_worlds[7]-changed_write;acc["write_error"]=max(acc["write_error"],float(error.abs().max()));acc["write_relative_num"]+=float(error.square().sum());acc["write_relative_den"]+=float((changed_write-write_worlds[0]).square().sum());acc["logit_error"]=max(acc["logit_error"],float((insert_logits[7]-independent_changed_write_logits).abs().max()));acc["examples"]+=len(changed_write)


def finalize(acc):
    state={n:acc["state_cross"][n]/max(acc["state_den"],1e-30) for n in GROUPS};baseline={n:acc["insert_base_cross"][n]/max(acc["insert_base_den"],1e-30) for n in GROUPS};changed={n:acc["insert_changed_cross"][n]/max(acc["insert_changed_den"],1e-30) for n in GROUPS};removal={n:acc["remove_changed_cross"][n]/max(acc["remove_changed_den"],1e-30) for n in GROUPS};forward_den=max(acc["forward_world_kl"][7],1e-30);target_den=max(acc["target_world_kl"][0],1e-30);forward=scalar_shapley([v/forward_den for v in acc["forward_world_kl"]]);target=scalar_shapley([1-v/target_den for v in acc["target_world_kl"]])
    return {"examples":acc["examples"],"state_projection":state,"allocations":{"baseline_fisher":baseline,"changed_fisher":changed,"removal_fisher":removal,"forward_kl":forward,"target_error_kl":target},"maximum_write_error":acc["write_error"],"relative_write_l2_error":float(np.sqrt(acc["write_relative_num"]/max(acc["write_relative_den"],1e-30))),"maximum_context_logit_disagreement":acc["logit_error"]}


@torch.inference_mode()
def evaluate(model,tokenizer,spec,position,config):
    rows=load_rows(spec);newline=tokenizer.encode("\n",add_special_tokens=False)[0];acc={layer:empty_accumulator() for layer in config["layers"]}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:config["sequence_length"]],dtype=np.int64);base_ids=torch.as_tensor(raw,device="cuda");keep=base_ids[:,position]!=newline
        if not bool(keep.any()):continue
        base_ids=base_ids[keep];changed_ids=base_ids.clone();changed_ids[:,position]=newline;base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids)
        for layer in config["layers"]:
            base=base_rows[layer];changed=changed_rows[layer];full_delta=changed["details"]["write"]-base["details"]["write"]
            deltas={};edited=torch.zeros_like(full_delta);edited[:,position]=full_delta[:,position];deltas["edited_position"]=edited;context=torch.zeros_like(full_delta);context[:,position+1:config["prediction_position"]]=full_delta[:,position+1:config["prediction_position"]];deltas["subsequent_context"]=context;final=torch.zeros_like(full_delta);final[:,config["prediction_position"]]=full_delta[:,config["prediction_position"]];deltas["final_position"]=final
            worlds=group_worlds(base["details"]["write"],deltas,GROUPS);insert=logits_from_attention_writes(model,layer,base,worlds);remove=logits_from_attention_writes(model,layer,changed,worlds);independent=logits_from_attention_writes(model,layer,base,changed["details"]["write"][None])[0];update(acc[layer],worlds,insert,remove,changed["details"]["write"],independent)
    return {str(layer):finalize(acc[layer]) for layer in config["layers"]}


def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");results={}
    for corpus,spec in config["evaluation"].items():
        results[corpus]={}
        for position in config["edit_positions"]:print("evaluating",corpus,position,flush=True);results[corpus][str(position)]=evaluate(model,tokenizer,spec,position,config)
    rows=[results[c][str(p)][str(layer)] for c in results for p in config["edit_positions"] for layer in config["layers"]]
    hypotheses={
      "V1_destination_partition_is_numerically_exact":{"supported":all(r["relative_write_l2_error"]<=1e-6 and r["maximum_context_logit_disagreement"]<=1e-4 for r in rows)},
      "V2_final_write_has_little_state_energy":{"supported":all(r["state_projection"]["final_position"]<=.05 for r in rows)},
      "V3_final_write_dominates_baseline_fisher_behavior":{"supported":all(r["allocations"]["baseline_fisher"]["final_position"]>=.50 for r in rows)},
      "V4_final_write_dominates_all_behavioral_metrics":{"supported":all(r["allocations"][metric]["final_position"]>=.50 for r in rows for metric in METRICS)},
      "V5_insertion_and_removal_preserve_final_dominance":{"supported":all(r["allocations"][metric]["final_position"]>=.50 for r in rows for metric in ("baseline_fisher","removal_fisher"))},
      "V6_state_behavior_inversion_is_large":{"supported":all(r["allocations"]["baseline_fisher"]["final_position"]-r["state_projection"]["final_position"]>=.40 for r in rows)},
      "V7_final_bottleneck_strengthens_with_depth":{"supported":all(results[c][str(p)]["10"]["allocations"]["baseline_fisher"]["final_position"]>=results[c][str(p)]["5"]["allocations"]["baseline_fisher"]["final_position"]-.05 for c in results for p in config["edit_positions"])},
      "V8_dominant_destination_transfers_ood":{"supported":all(len({max(GROUPS,key=lambda n:abs(results[c][str(p)][str(layer)]["allocations"][metric][n])) for c in results})==1 for p in config["edit_positions"] for layer in config["layers"] for metric in METRICS)},
    }
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    layers=np.array(config["layers"]);fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for position in config["edit_positions"]:
        axes[0].plot(layers,[np.mean([results[c][str(position)][str(layer)]["state_projection"]["final_position"] for c in results]) for layer in layers],marker="o",label=f"distance {127-position}");axes[1].plot(layers,[np.mean([results[c][str(position)][str(layer)]["allocations"]["baseline_fisher"]["final_position"] for c in results]) for layer in layers],marker="o",label=f"distance {127-position}")
    for metric in METRICS:
        axes[2].plot(layers,[np.mean([results[c]["95"][str(layer)]["allocations"][metric]["final_position"] for c in results]) for layer in layers],marker="o",label=metric)
    axes[0].set_title("Final-write state projection");axes[1].set_title("Final-write behavioral allocation");axes[2].set_title("Position-95 metric red team")
    for axis in axes:axis.set_xlabel("layer");axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Destination metric and distance red team","","At edit positions 63, 95, and 119, every recurrent-backbone attention delta is split into the edited destination itself, all subsequent context destinations, and the final prediction destination. The three exact write parts are evaluated under base-context insertion, changed-context removal, two Fisher references, and two exact-KL cooperative games.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Destination red team](bilin12_attention_destination_metric_redteam.png)","","| Corpus | Position | Layer | Final state | Final base-Fisher / changed-Fisher / removal | Final forward-KL / target-error-KL |","|---|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for position in config["edit_positions"]:
            for layer in config["layers"]:
                row=results[corpus][str(position)][str(layer)];a=row["allocations"];lines.append(f"| {corpus} | {position} | {layer} | {row['state_projection']['final_position']:.3f} | {a['baseline_fisher']['final_position']:.3f} / {a['changed_fisher']['final_position']:.3f} / {a['removal_fisher']['final_position']:.3f} | {a['forward_kl']['final_position']:.3f} / {a['target_error_kl']['final_position']:.3f} |")
    lines += ["","A destination allocation answers which part of one layer's native attention-message change controls the selected final distribution. It does not imply that the final destination locally stores the original token: its query, keys, and values already depend on the entire prior computation."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)


if __name__=="__main__":main()
