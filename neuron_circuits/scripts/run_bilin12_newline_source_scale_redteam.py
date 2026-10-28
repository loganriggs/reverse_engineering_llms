#!/usr/bin/env python3
"""Separate effect scale from source geometry in newline source programs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes,trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT

CONFIG=Path("configs/bilin12_newline_source_scale_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin12_newline_source_scale_redteam.json"
REPORT=TENSOR_ROOT/"BILIN12_NEWLINE_SOURCE_SCALE_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin12_newline_source_scale_redteam.png"

def empty():return {"examples":0,"target":0.,"error":0.}
def finish(row):return {"examples":row["examples"],"target_energy":row["target"],"fisher_fidelity":1-row["error"]/max(row["target"],1e-30)}

@torch.inference_mode()
def evaluate(model,tokenizer,rows,edit_position,config):
    newline=tokenizer.encode("\n",add_special_tokens=False)[0];position=config["prediction_position"];acc={str(layer):{str(scale):{method:empty() for method in config["methods"]} for scale in config["scales"]} for layer in config["layers"]}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:config["sequence_length"]],dtype=np.int64);base_ids=torch.as_tensor(raw,device="cuda",dtype=torch.long);keep=base_ids[:,edit_position]!=newline
        if not bool(keep.any()):continue
        base_ids=base_ids[keep];changed_ids=base_ids.clone();changed_ids[:,edit_position]=newline;base_rows,_,_=trace(model,base_ids);changed_rows,_,_=trace(model,changed_ids)
        for layer in config["layers"]:
            key=str(layer);base=base_rows[layer];changed=changed_rows[layer];source_delta=(source_write_contributions(model.blocks[layer],changed["details"],position)-source_write_contributions(model.blocks[layer],base["details"],position)).float();direct=(changed["details"]["write"][:,position]-base["details"]["write"][:,position]).float();pattern=(changed["details"]["pattern"][:,:,position]-base["details"]["pattern"][:,:,position]).abs().sum(1)
            state_order=source_delta.square().sum(-1).argsort(-1,descending=True);pattern_order=pattern.argsort(-1,descending=True);recency=torch.arange(config["sequence_length"]-1,-1,-1,device="cuda")[None].repeat(len(base_ids),1);orders={"state16":state_order[:,:16],"state32":state_order[:,:32],"pattern32":pattern_order[:,:32],"recency32":recency[:,:32]};approximations={}
            for method,indices in orders.items():mask=torch.zeros(len(base_ids),config["sequence_length"],device="cuda");mask.scatter_(1,indices,1);approximations[method]=(mask[:,:,None]*source_delta).sum(1)
            for scale in config["scales"]:
                worlds=[];target_write=base["details"]["write"].clone();target_write[:,position]+=(scale*direct).to(target_write.dtype);worlds.append(target_write)
                for method in config["methods"]:
                    write=base["details"]["write"].clone();write[:,position]+=(scale*approximations[method]).to(write.dtype);worlds.append(write)
                base_logits=logits_from_attention_writes(model,layer,base,base["details"]["write"].unsqueeze(0))[0];logits=logits_from_attention_writes(model,layer,base,torch.stack(worlds));target_delta=logits[0]-base_logits;target,_,_,_=fisher_terms(base_logits,target_delta,target_delta)
                for index,method in enumerate(config["methods"],1):
                    _,_,_,error=fisher_terms(base_logits,target_delta,logits[index]-base_logits);row=acc[key][str(scale)][method];row["examples"]+=len(base_ids);row["target"]+=float(target.sum());row["error"]+=float(error.sum())
    return {layer:{scale:{method:finish(row) for method,row in methods.items()} for scale,methods in scales.items()} for layer,scales in acc.items()}

def main():
    config=json.loads(CONFIG.read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");results={}
    for corpus,spec in config["evaluation"].items():
        rows=load_rows(spec);results[corpus]={}
        for position in config["edit_positions"]:print("evaluating",corpus,position,flush=True);results[corpus][str(position)]=evaluate(model,tokenizer,rows,position,config)
    def vals(scale,method):return [results[c][str(p)][str(layer)][str(scale)][method]["fisher_fidelity"] for c in results for p in config["edit_positions"] for layer in config["layers"]]
    def global_fidelity(scale,method):
        rows=[results[c][str(p)][str(layer)][str(scale)][method] for c in results for p in config["edit_positions"] for layer in config["layers"]];target=sum(r["target_energy"] for r in rows);error=sum((1-r["fisher_fidelity"])*r["target_energy"] for r in rows);return 1-error/max(target,1e-30)
    energy={str(scale):sum(results[c][str(p)][str(layer)][str(scale)]["state32"]["target_energy"] for c in results for p in config["edit_positions"] for layer in config["layers"]) for scale in config["scales"]};small=[energy[str(scale)]/scale**2 for scale in config["scales"][:3]];cv=float(np.std(small)/np.mean(small));semantic=json.loads(Path(config["semantic_reference"]).read_text());semantic_rows=semantic["overall"];target=sum(r["state_norm/32"]["target_energy"] for r in semantic_rows.values());error=sum((1-r["state_norm/32"]["fisher_fidelity"])*r["state_norm/32"]["target_energy"] for r in semantic_rows.values());semantic_fidelity=1-error/target;frontier={method:[global_fidelity(scale,method) for scale in config["scales"]] for method in config["methods"]};a1=frontier["state32"][0]>=.97;a9=semantic_fidelity-frontier["state32"][0]>=.02
    hypotheses={"A1_small_newline_edits_match_semantic_state32_fidelity":{"supported":a1,"small_scale":frontier["state32"][0]},"A2_state32_fidelity_improves_as_scale_shrinks":{"supported":all(b<=a+1e-6 for a,b in zip(frontier["state32"][:-1],frontier["state32"][1:]))},"A3_small_scale_repairs_the_worst_conditions":{"supported":min(vals(.125,"state32"))-min(vals(1.0,"state32"))>=.10,"improvement":min(vals(.125,"state32"))-min(vals(1.0,"state32"))},"A4_pattern32_improves_as_scale_shrinks":{"supported":all(b<=a+1e-6 for a,b in zip(frontier["pattern32"][:-1],frontier["pattern32"][1:]))},"A5_fisher_target_energy_is_locally_quadratic":{"supported":cv<=.10,"coefficient_of_variation":cv},"A6_state_rule_still_beats_recency_at_small_scale":{"supported":frontier["state32"][0]-frontier["recency32"][0]>=.20,"advantage":frontier["state32"][0]-frontier["recency32"][0]},"A7_state16_remains_less_robust_than_state32":{"supported":all(frontier["state16"][i]<frontier["state32"][i] for i in range(len(config["scales"])))},"A8_large_scale_exposes_nonlinear_metric_failure":{"supported":min(min(vals(1.5,m)) for m in config["methods"])<0},"A9_scale_does_not_fully_explain_semantic_transfer":{"supported":a9,"semantic_fidelity":semantic_fidelity,"small_newline_fidelity":frontier["state32"][0],"gap":semantic_fidelity-frontier["state32"][0]},"A10_source_geometry_and_effect_scale_both_matter":{"supported":a1 and a9}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"frontier":frontier,"target_energy":energy,"results":results},indent=2,allow_nan=False));fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    for method in config["methods"]:axes[0].plot(config["scales"],frontier[method],marker="o",label=method);axes[1].plot(config["scales"],[min(vals(scale,method)) for scale in config["scales"]],marker="o",label=method)
    axes[0].axhline(semantic_fidelity,color="black",ls="--",lw=.8,label="semantic state32");axes[0].set_title("Global fidelity versus scale");axes[1].set_title("Worst-condition fidelity");axes[2].plot(config["scales"],[energy[str(scale)]/scale**2 for scale in config["scales"]],marker="o");axes[2].set_title("Target energy / scale²")
    for axis in axes:axis.set_xlabel("intervention scale");axis.grid(alpha=.2);axis.legend(fontsize=7) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_ylabel("Fisher fidelity");axes[1].set_ylabel("Fisher fidelity");fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Newline source-program scale red team","","Fresh newline counterfactuals keep the exact source-edge ranking fixed while both the full final-destination message and each truncated approximation are scaled from 0.125x to 1.5x before suffix recomputation. This separates source geometry from finite-effect nonlinearity and compares the small-scale limit with the semantic-edit transfer result.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Scale red team](bilin12_newline_source_scale_redteam.png)","","| Scale | State16 | State32 | Pattern32 | Recency32 | State32 minimum |","|---:|---:|---:|---:|---:|---:|"]
    for i,scale in enumerate(config["scales"]):lines.append(f"| {scale:.3f} | {frontier['state16'][i]:.3f} | {frontier['state32'][i]:.3f} | {frontier['pattern32'][i]:.3f} | {frontier['recency32'][i]:.3f} | {min(vals(scale,'state32')):.3f} |")
    lines += ["",f"Semantic state-norm top-32 reference: **{semantic_fidelity:.3f}**. Small-scale newline state32: **{frontier['state32'][0]:.3f}**. Gap: **{semantic_fidelity-frontier['state32'][0]:.3f}**. Local quadratic-energy coefficient of variation: **{cv:.3f}**."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)

if __name__=="__main__":main()
