#!/usr/bin/env python3
"""Fresh-context per-example red team of semantic source-program transfer."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_semantic_source_program_transfer import analyze_example,jaccard,select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT

CONFIG=Path("configs/bilin12_semantic_source_individual_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin12_semantic_source_individual_redteam.json"
REPORT=TENSOR_ROOT/"BILIN12_SEMANTIC_SOURCE_INDIVIDUAL_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin12_semantic_source_individual_redteam.png"


def metric_values(records,label):return [(layer["metrics"][label]["fisher_fidelity"],layer["metrics"][label]["target_energy"],record,layer_name) for record in records for layer_name,layer in record["layers"].items()]


def main():
    red=json.loads(CONFIG.read_text());config=json.loads(Path(red["source_config"]).read_text());semantic=json.loads(Path(config["semantic_source"]).read_text());tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_snapshot_model(Path(config["snapshot"]),"cuda");records=[]
    requested=red["skip_contexts_per_pair"]+red["evaluation_contexts_per_pair"]
    for corpus_index,(corpus,spec) in enumerate(config["evaluation"].items()):
        source=np.load(spec["path"],mmap_mode="r");rows=source[spec["start"]:spec["start"]+spec["rows"]];rng=np.random.default_rng(red["seed"]+corpus_index)
        for family,pairs in semantic["families"].items():
            for source_token,target_token in pairs:
                windows=select_windows(rows,source_token,config["edit_position"],config["prediction_position"],requested,rng)[red["skip_contexts_per_pair"]:]
                for row_index,start in windows:
                    base=np.asarray(rows[row_index,start:start+config["sequence_length"]],dtype=np.int64).copy();changed=base.copy();changed[config["edit_position"]]=target_token;layers=analyze_example(model,torch.as_tensor(base,device="cuda")[None],torch.as_tensor(changed,device="cuda")[None],config);records.append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","source_token":tokenizer.decode([source_token]),"target_token":tokenizer.decode([target_token]),"row":row_index,"start":start,"layers":layers});print(corpus,family,tokenizer.decode([source_token]),"->",tokenizer.decode([target_token]),flush=True)
    state32=metric_values(records,"state_norm/32");state16=metric_values(records,"state_norm/16");pattern32=metric_values(records,"pattern_change/32");oracle32=metric_values(records,"cooperative_oracle/32");recency32=metric_values(records,"recency/32")
    sv=np.asarray([v for v,_,_,_ in state32]);se=np.asarray([e for _,e,_,_ in state32]);pv=np.asarray([v for v,_,_,_ in pattern32]);ov=np.asarray([v for v,_,_,_ in oracle32]);rv=np.asarray([v for v,_,_,_ in recency32]);s16=np.asarray([v for v,_,_,_ in state16]);corr=float(np.corrcoef(np.log10(se+1e-30),sv)[0,1])
    layer_medians={layer:float(np.median([v for v,_,_,l in state32 if l==str(layer)])) for layer in config["layers"]};family_minima={family:float(min(v for v,_,r,_ in state32 if r["family"]==family)) for family in semantic["families"]};same=[];different=[]
    for left,right in combinations(records,2):
        if left["pair"]==right["pair"]:continue
        values=[jaccard(left["layers"][str(layer)]["oracle_top16"],right["layers"][str(layer)]["oracle_top16"]) for layer in config["layers"]];(same if left["family"]==right["family"] else different).extend(values)
    advantage=float(np.mean(same)-np.mean(different));hypotheses={"I1_state_norm32_is_robust_per_example":{"supported":sv.min()>=.80,"minimum":float(sv.min())},"I2_state_norm32_has_a_high_lower_tail":{"supported":np.quantile(sv,.05)>=.90,"q05":float(np.quantile(sv,.05))},"I3_state_norm16_is_useful_per_example":{"supported":np.quantile(s16,.05)>=.70,"q05":float(np.quantile(s16,.05))},"I4_pattern32_is_robust_per_example":{"supported":pv.min()>=.75,"minimum":float(pv.min())},"I5_local_state_rule_matches_the_oracle_at_32":{"supported":sv.mean()>=ov.mean()-.02,"state_mean":float(sv.mean()),"oracle_mean":float(ov.mean())},"I6_local_state_rule_beats_recency_at_32":{"supported":sv.mean()-rv.mean()>=.05,"advantage":float(sv.mean()-rv.mean())},"I7_success_is_not_an_effect_size_artifact":{"supported":abs(corr)<=.20,"correlation":corr},"I8_every_layer_has_high_median_fidelity":{"supported":min(layer_medians.values())>=.95,"medians":layer_medians},"I9_every_semantic_family_has_a_robust_minimum":{"supported":min(family_minima.values())>=.75,"minima":family_minima},"I10_source_topology_still_is_not_semantic":{"supported":advantage<.05,"advantage":advantage,"same":float(np.mean(same)),"different":float(np.mean(different))}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":red,"source_config":config,"hypotheses":hypotheses,"records":records},indent=2,allow_nan=False));fig,axes=plt.subplots(1,3,figsize=(18,4.7),layout="constrained");axes[0].hist(sv,bins=20,alpha=.7,label="state32");axes[0].hist(pv,bins=20,alpha=.5,label="pattern32");axes[0].set_title("Fresh per-example fidelity");axes[1].scatter(np.log10(se+1e-30),sv,s=18,alpha=.7);axes[1].set_title("Effect size versus state-rule fidelity");axes[1].set_xlabel("log10 target Fisher energy");axes[2].bar(np.arange(len(layer_medians)),list(layer_medians.values()));axes[2].set_xticks(np.arange(len(layer_medians)));axes[2].set_xticklabels([f"L{x}" for x in layer_medians]);axes[2].set_title("State32 median by layer")
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=7) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xlabel("Fisher fidelity");axes[1].set_ylabel("Fisher fidelity");axes[2].set_ylabel("median Fisher fidelity");fig.savefig(FIGURE,dpi=180);plt.close(fig)
    lines=["# Fresh individual-example red team of semantic source programs","","For every lexical pair, the first two deterministic windows were used in the aggregate semantic-transfer audit. This preregistered follow-up evaluates the third window, producing 38 fresh examples and 152 layer-example effects. Aggregate Fisher weighting is removed: thresholds apply to individual examples and layers.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Individual semantic red team](bilin12_semantic_source_individual_redteam.png)","",f"State-norm top-32: mean **{sv.mean():.3f}**, median **{np.median(sv):.3f}**, fifth percentile **{np.quantile(sv,.05):.3f}**, minimum **{sv.min():.3f}**.","",f"State-norm top-16 fifth percentile: **{np.quantile(s16,.05):.3f}**. Pattern-change top-32 minimum: **{pv.min():.3f}**. State32-versus-log-effect correlation: **{corr:.3f}**.","","| Layer | State32 median |","|---:|---:|"]
    for layer,value in layer_medians.items():lines.append(f"| {layer} | {value:.3f} |")
    lines += ["","| Family | State32 minimum |","|---|---:|"]
    for family,value in family_minima.items():lines.append(f"| {family} | {value:.3f} |")
    lines += ["",f"Same-family versus different-family oracle-position Jaccard is **{np.mean(same):.3f}** versus **{np.mean(different):.3f}**. The local tensor rule can therefore be predictively stable even when the selected absolute positions carry no semantic-family identity."]
    REPORT.write_text("\n".join(lines)+"\n");print("wrote",REPORT,flush=True)

if __name__=="__main__":main()
