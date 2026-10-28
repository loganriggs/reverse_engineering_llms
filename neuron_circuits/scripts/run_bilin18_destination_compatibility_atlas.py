#!/usr/bin/env python3
"""Ask whether independent tensor geometry predicts destination compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank, intervention_logits
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import spearman


CONFIG=Path("configs/bilin18_destination_compatibility_atlas.json")
GEOMETRY=TENSOR_ROOT/"bilin18_bus_aware_function_templates.pt"
OUTPUT=TENSOR_ROOT/"bilin18_destination_compatibility_atlas.json"
REPORT=TENSOR_ROOT/"BILIN18_DESTINATION_COMPATIBILITY_ATLAS.md"


def main():
    config=json.loads(CONFIG.read_text());geometry=torch.load(GEOMETRY,map_location="cpu",weights_only=True)
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    bank,gram,norms=build_function_bank(model,[0,1,2,3,4,5]);labels=[row["label"] for row in bank]
    target_label=f"L{config['target'][0]}H{config['target'][1]}";partner_label=f"L{config['partner'][0]}H{config['partner'][1]}"
    target=labels.index(target_label);partner=labels.index(partner_label);partner_scale,controls,cosine=complete_controls(gram,norms,target,partner)
    corpus=np.load(config["corpus"],mmap_mode="r");raw=np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"]+config["evaluation_rows"],:config["sequence_length"]+1],dtype=np.int64)
    halves={"a":raw[:len(raw)//2],"b":raw[len(raw)//2:]};positions=torch.as_tensor(config["positions"],device="cuda");kl_by_half={}
    for half,panel in halves.items():
        print("destination compatibility",half,flush=True);totals=torch.zeros(3+len(controls),dtype=torch.float64,device="cuda");sites=0
        for start in range(0,len(panel),config["evaluation_batch_size"]):
            ids=torch.as_tensor(panel[start:start+config["evaluation_batch_size"],:config["sequence_length"]],device="cuda")
            logits,_=intervention_logits(model,ids,bank[target],bank[partner],partner_scale,controls,bank,positions,config["intervention_scale"])
            logp=F.log_softmax(logits[0].float(),-1)
            for variant in range(2,len(logits)):
                logq=F.log_softmax(logits[variant].float(),-1);totals[variant]+=(logp.exp()*(logp-logq)).sum(-1).clamp_min(0).double().sum()
            sites+=logits.shape[1]*logits.shape[2]
        means=(totals/sites).cpu().numpy();values={partner_label:float(means[2])}
        for control,value in zip(controls,means[3:]):values[labels[control["other"]]]=float(value)
        kl_by_half[half]=values
    label_index={label:index for index,label in enumerate(geometry["labels"])};mode=geometry["output_modes"].numpy();bus=geometry["bus_cosine"].numpy();nat_a=geometry["natural_a"].numpy();nat_b=geometry["natural_b"].numpy();ti=label_index[target_label]
    candidates=[label for label in labels if label!=target_label];rows=[]
    for label in candidates:
        index=label_index[label];full=abs(float(bus[ti,index]));natural=min(abs(float(nat_a[ti,index])),abs(float(nat_b[ti,index])));output=abs(float(mode[ti]@mode[index]));joint=full*natural
        rows.append({"candidate":label,"full_operator_abs_cosine":full,"replicated_natural_abs_cosine":natural,"output_verb_abs_cosine":output,"joint_geometry_score":joint,"kl_a":kl_by_half["a"][label],"kl_b":kl_by_half["b"][label]})
    kl_a=np.asarray([row["kl_a"] for row in rows]);kl_b=np.asarray([row["kl_b"] for row in rows]);ranking_stability=spearman(kl_a,kl_b)
    correlations={}
    for feature in ("joint_geometry_score","replicated_natural_abs_cosine","output_verb_abs_cosine","full_operator_abs_cosine"):
        values=np.asarray([row[feature] for row in rows]);correlations[feature]={"a":spearman(values,-kl_a),"b":spearman(values,-kl_b)}
    partner_row=next(row for row in rows if row["candidate"]==partner_label);rank_a=int(np.argsort(kl_a).tolist().index(rows.index(partner_row))+1);rank_b=int(np.argsort(kl_b).tolist().index(rows.index(partner_row))+1)
    hypotheses={
        "A1_compatibility_ranking_is_split_stable":{"supported":ranking_stability>=.80,"spearman":ranking_stability},
        "A2_locked_joint_geometry_predicts_compatibility":{"supported":all(value>=.30 for value in correlations["joint_geometry_score"].values()),"correlations":correlations["joint_geometry_score"]},
        "A3_natural_response_similarity_predicts_compatibility":{"supported":all(value>=.30 for value in correlations["replicated_natural_abs_cosine"].values()),"correlations":correlations["replicated_natural_abs_cosine"]},
        "A4_output_direction_similarity_predicts_compatibility":{"supported":all(value>=.30 for value in correlations["output_verb_abs_cosine"].values()),"correlations":correlations["output_verb_abs_cosine"]},
        "A5_geometry_selected_partner_is_causally_best":{"supported":rank_a==1 and rank_b==1,"ranks":{"a":rank_a,"b":rank_b}},
    }
    for value in hypotheses.values():value["supported"]=bool(value["supported"])
    rows.sort(key=lambda row:(row["kl_a"]+row["kl_b"])/2)
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"correlations":correlations,"candidates":rows},indent=2,allow_nan=False))
    lines=["# Destination-aware operator compatibility atlas","",f"Every non-target early operator is converted into an exact norm-and-angle-matched replacement for {target_label} under its native router. Finite KL is measured at scale {config['intervention_scale']} on two held-out halves. Candidate scores are frozen from the earlier bus-aware weight and natural-response panels; no evaluation KL enters them.","","## Locked hypotheses",""]
    for name,value in hypotheses.items():lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["",f"Finite compatibility rank correlation across halves: `{ranking_stability:.3f}`.","","| Independent feature | Spearman with −KL A/B |","|---|---:|"]
    for feature,value in correlations.items():lines.append(f"| {feature} | {value['a']:.3f} / {value['b']:.3f} |")
    lines += ["","Best finite replacements:",""]
    for row in rows[:12]:lines.append(f"- {row['candidate']}: KL `{row['kl_a']:.3e}/{row['kl_b']:.3e}`, joint geometry `{row['joint_geometry_score']:.4f}`, natural `{row['replicated_natural_abs_cosine']:.3f}`, output `{row['output_verb_abs_cosine']:.3f}`")
    lines += ["","## Guardrails","","The candidate score chose L2H5 before this finite panel, but its formula was motivated by prior results and is not a universal compatibility law. All replacements retain the target router and destination. A correlation failure would mean the successful pair is an outlier, not that its finite advantage is false; success would nominate a prospective metric for other targets that still requires fresh-target validation."]
    REPORT.write_text("\n".join(lines)+"\n");print(json.dumps({"hypotheses":hypotheses,"correlations":correlations},indent=2));print(REPORT)


if __name__=="__main__":main()
