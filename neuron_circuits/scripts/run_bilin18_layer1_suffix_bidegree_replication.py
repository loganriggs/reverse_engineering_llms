#!/usr/bin/env python3
"""Full finite replication of the MLP×attention semantic backward program."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import bidegree_program, metrics
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_bidegree_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_bidegree_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_BIDEGREE_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_bidegree_replication.png"


def aggregate(records, method):
    shifts=np.asarray([row["finite_shifts"][method] for row in records]); return {"directional_fraction":float(np.mean(shifts>0)),"mean_shift":float(shifts.mean()),"mean_absolute_shift":float(np.abs(shifts).mean())}


def main():
    config=json.loads(CONFIG.read_text());causal=json.loads(Path(config["source_config"]).read_text());old=json.loads(Path(config["source_results"]).read_text());source=json.loads(Path(causal["source_config"]).read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]);positive=flags.sum(0).astype(np.float32);negative=len(flags)-positive;contrast=torch.as_tensor(flags/positive.clip(1)[None]-(1-flags)/negative.clip(1)[None],device="cuda",dtype=torch.float32)
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    corpus=np.load(source["corpus"],mmap_mode="r");heldout_start=source["start_row"]+source["discovery_rows"];k=config["support_rank"];records=[]
    for site,row in enumerate(old["records"]):
        feature=FEATURES.index(row["feature"]);vector=contrast[:,feature];donor_raw=np.asarray(corpus[heldout_start+row["donor_row"],:source["sequence_length"]],dtype=np.int64);recipient_raw=np.asarray(corpus[heldout_start+row["recipient_row"],:source["sequence_length"]],dtype=np.int64);donor_ids=torch.as_tensor(donor_raw,device="cuda")[None];recipient_ids=torch.as_tensor(recipient_raw,device="cuda")[None]
        with torch.no_grad():donor_trace=forward_layers_native(model,donor_ids,1);recipient_trace=forward_layers_native(model,recipient_ids,17)
        delta=product_at(model,donor_trace,1,row["donor_position"])-product_at(model,recipient_trace,1,row["recipient_position"]);exact=semantic_gradient(model,recipient_trace,1,row["recipient_position"],vector);grid=bidegree_program(model,recipient_trace,row["recipient_position"],vector,config["maximum_total_degree"])
        choose=lambda predicate:sum((value for key,value in grid.items() if predicate(*key)),torch.zeros_like(exact));predictions={"mlp_only":choose(lambda m,a:a==0),"axes_union":choose(lambda m,a:m==0 or a==0),"rectangle4":choose(lambda m,a:m<=4 and a<=4),"rectangle8":choose(lambda m,a:m<=8 and a<=8),"rectangle12":choose(lambda m,a:m<=12 and a<=12),"full":choose(lambda m,a:True),"exact":exact};supports={name:(prediction*delta).topk(k).indices for name,prediction in predictions.items()};deltas={"base":torch.zeros_like(delta),**{name:selected_delta(delta,support) for name,support in supports.items()}}
        with torch.no_grad():logits=logits_for_deltas(model,1,recipient_trace[1],deltas,row["recipient_position"])
        base=class_score(logits["base"],vector);finite={name:class_score(value,vector)-base for name,value in logits.items() if name!="base"};records.append({"feature":row["feature"],"selection":row["selection"],"metrics":{name:metrics(prediction,exact,delta,k) for name,prediction in predictions.items()},"finite_shifts":finite})
        if (site+1)%8==0:print("bidegree replication",site+1,"/",len(old["records"]),flush=True)
    methods=list(records[0]["metrics"]);summaries={}
    for method in methods:
        values=[row["metrics"][method] for row in records];summaries[method]={metric+"_median":float(np.median([value[metric] for value in values])) for metric in ("gradient_cosine","gradient_relative_error","contribution_cosine","true_leverage_ratio")};summaries[method]["gradient_cosine_q10"]=float(np.quantile([value["gradient_cosine"] for value in values],.1));summaries[method]["finite"]=aggregate(records,method)
    exact_finite=summaries["exact"]["finite"];per_feature8={feature:{"gradient_cosine_median":float(np.median([row["metrics"]["rectangle8"]["gradient_cosine"] for row in records if row["feature"]==feature])),"finite_directional_fraction":float(np.mean([row["finite_shifts"]["rectangle8"]>0 for row in records if row["feature"]==feature]))} for feature in causal["features"]};r4=summaries["rectangle4"];r8=summaries["rectangle8"];axes=summaries["axes_union"];full=summaries["full"]
    hypotheses={"J0_full_bidegree_recomposition_is_exact":{"supported":all(row["metrics"]["full"]["gradient_cosine"]>=.999 and row["metrics"]["full"]["gradient_relative_error"]<=.02 for row in records)},"J1_rectangle8_replication":{"supported":r8["gradient_cosine_median"]>=.90 and r8["gradient_cosine_q10"]>=.75},"J2_rectangle8_finite_direction":{"supported":r8["finite"]["directional_fraction"]>=.95},"J3_rectangle8_causal_strength":{"supported":r8["finite"]["mean_shift"]>=.80*exact_finite["mean_shift"]},"J4_cross_paths_beat_axes":{"supported":r8["gradient_cosine_median"]-axes["gradient_cosine_median"]>=.15 and (r8["finite"]["mean_shift"]-axes["finite"]["mean_shift"])/exact_finite["mean_shift"]>=.10},"J5_rectangle8_transfers_across_features":{"supported":all(value["gradient_cosine_median"]>=.75 and value["finite_directional_fraction"]>=.75 for value in per_feature8.values())},"J6_rectangle4_is_insufficient_control":{"supported":r4["gradient_cosine_median"]<.80 or r4["finite"]["mean_shift"]<.70*exact_finite["mean_shift"]}}
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"summaries":summaries,"per_feature_rectangle8":per_feature8,"records":records},indent=2,allow_nan=False));lines=["# Full finite replication of the MLP×attention bidegree program","","All 96 frozen same-token semantic pairs are evaluated. Each bidegree truncation selects its own top-64 positive-contribution channels, which are patched through the complete nonlinear suffix.","","## Locked hypotheses",""]
    for name,value in hypotheses.items():lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["","![Bidegree replication](bilin18_layer1_suffix_bidegree_replication.png)","","| Program | Grad. cosine / Q10 | Contribution cosine | Leverage | Finite direction | Mean shift ratio |","|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        value=summaries[method];lines.append(f"| {method} | {value['gradient_cosine_median']:.3f}/{value['gradient_cosine_q10']:.3f} | {value['contribution_cosine_median']:.3f} | {value['true_leverage_ratio_median']:.3f} | {value['finite']['directional_fraction']:.3f} | {value['finite']['mean_shift']/exact_finite['mean_shift']:.3f} |")
    lines += ["","## Guardrail","","The rectangle is an operator-interaction grammar. It is not a sparse edge list and does not imply that attention has an independent semantic direction; the axes-versus-cross-path comparison tests that distinction."];REPORT.write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout="constrained");plot_methods=("mlp_only","axes_union","rectangle4","rectangle8","rectangle12","full");ax[0].bar(range(len(plot_methods)),[summaries[m]["gradient_cosine_median"] for m in plot_methods]);ax[0].set_xticks(range(len(plot_methods)),plot_methods,rotation=30,ha="right");ax[0].set(ylabel="Median gradient cosine",title="Cross-family operator composition");ax[0].grid(axis="y",alpha=.2);ax[1].bar(range(len(plot_methods)),[summaries[m]["finite"]["mean_shift"]/exact_finite["mean_shift"] for m in plot_methods]);ax[1].set_xticks(range(len(plot_methods)),plot_methods,rotation=30,ha="right");ax[1].axhline(1,color="gray",linestyle=":");ax[1].set(ylabel="Mean finite shift / exact",title="Causal strength of the selected coalition");ax[1].grid(axis="y",alpha=.2);fig.savefig(FIGURE,dpi=180);plt.close(fig);print(json.dumps({"hypotheses":hypotheses,"summaries":summaries,"per_feature_rectangle8":per_feature8},indent=2));print(REPORT)


if __name__=="__main__":main()
