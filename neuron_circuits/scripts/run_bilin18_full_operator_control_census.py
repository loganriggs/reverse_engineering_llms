#!/usr/bin/env python3
"""Complete matched-control census for directional full-operator compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_full_operator_transplant import build_function_bank, intervention_logits
from run_bilin18_output_verb_transplant import checkpoint_logits
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_full_operator_control_census.json")
OUTPUT = TENSOR_ROOT / "bilin18_full_operator_control_census.json"
REPORT = TENSOR_ROOT / "BILIN18_FULL_OPERATOR_CONTROL_CENSUS.md"


def complete_controls(gram, norms, target, partner):
    signed = gram[target, partner] / (norms[target] * norms[partner])
    sign = 1.0 if signed >= 0 else -1.0; cosine = abs(float(signed))
    controls = []
    for other in range(len(norms)):
        if other in (target, partner): continue
        projection = gram[target, other] / gram[target, target]
        orthogonal_norm = np.sqrt(max(gram[other, other] - gram[target, other] ** 2 / gram[target, target], 1e-30))
        beta = np.sqrt(max(1 - cosine ** 2, 0)) * norms[target] / orthogonal_norm
        controls.append({"other": other, "target_coefficient": float(cosine - beta * projection), "other_coefficient": float(beta)})
    return float(sign * norms[target] / norms[partner]), controls, cosine


def main():
    config=json.loads(CONFIG.read_text()); model=load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    bank,gram,norms=build_function_bank(model,[0,1,2,3,4,5]); labels=[row["label"] for row in bank]
    pair_labels=[f"L{l}H{h}" for l,h in config["pair"]]; pair=[labels.index(label) for label in pair_labels]
    programs=[]
    for d in range(2):
        target,partner=pair[d],pair[1-d]; scale,controls,cosine=complete_controls(gram,norms,target,partner)
        programs.append({"name":f"{pair_labels[d]}_to_{pair_labels[1-d]}","target":target,"partner":partner,"scale":scale,"controls":controls,"cosine":cosine})
    corpus=np.load(config["corpus"],mmap_mode="r"); raw=np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"]+config["evaluation_rows"],:config["sequence_length"]+1],dtype=np.int64)
    halves={"a":raw[:len(raw)//2],"b":raw[len(raw)//2:]}; positions=torch.as_tensor(config["positions"],device="cuda")
    validation_ids=torch.as_tensor(raw[:1,:config["sequence_length"]],device="cuda"); reference=checkpoint_logits(model,validation_ids,positions)
    maximum_native_error=0.0; maximum_factor_error=0.0; results={}
    for program in programs:
        check,error=intervention_logits(model,validation_ids,bank[program["target"]],bank[program["partner"]],program["scale"],program["controls"],bank,positions)
        maximum_native_error=max(maximum_native_error,float((check[0]-reference).abs().max())); maximum_factor_error=max(maximum_factor_error,error)
        for half,panel in halves.items():
            print("complete operator controls",program["name"],half,flush=True)
            totals=torch.zeros(3+len(program["controls"]),dtype=torch.float64,device="cuda");sites=0
            for start in range(0,len(panel),config["evaluation_batch_size"]):
                ids=torch.as_tensor(panel[start:start+config["evaluation_batch_size"],:config["sequence_length"]],device="cuda")
                logits,error=intervention_logits(model,ids,bank[program["target"]],bank[program["partner"]],program["scale"],program["controls"],bank,positions); maximum_factor_error=max(maximum_factor_error,error)
                logp=F.log_softmax(logits[0].float(),-1)
                for variant in range(1,len(logits)):
                    logq=F.log_softmax(logits[variant].float(),-1);totals[variant]+=(logp.exp()*(logp-logq)).sum(-1).clamp_min(0).double().sum()
                sites+=logits.shape[1]*logits.shape[2]
            means=(totals/sites).cpu().numpy();controls=means[3:]; percentile=float(np.mean(controls<=means[2]))
            results[f"{program['name']}_{half}"]={"operator_cosine":program["cosine"],"delete_kl":float(means[1]),"partner_kl":float(means[2]),"control_median":float(np.median(controls)),"control_p10":float(np.quantile(controls,.1)),"control_p90":float(np.quantile(controls,.9)),"partner_percentile":percentile,"partner_control_ratio":float(means[2]/max(np.median(controls),1e-30)),"control_kl":controls.tolist()}
    forward=programs[1] if programs[1]["name"].startswith("L3H8") else programs[0]
    reverse=programs[0] if forward is programs[1] else programs[1]
    c1=all(results[f"{forward['name']}_{h}"]["partner_percentile"]<=.10 for h in halves)
    c2=all(results[f"{reverse['name']}_{h}"]["partner_percentile"]>=.50 for h in halves)
    gaps={h:results[f"{reverse['name']}_{h}"]["partner_percentile"]-results[f"{forward['name']}_{h}"]["partner_percentile"] for h in halves}
    c3=all(value>=.30 for value in gaps.values())
    c4=all(results[f"{p['name']}_{h}"]["partner_kl"]<results[f"{p['name']}_{h}"]["delete_kl"] for p in programs for h in halves)
    c5=all(abs(results[f"{p['name']}_a"]["partner_percentile"]-results[f"{p['name']}_b"]["partner_percentile"])<=.15 for p in programs)
    hypotheses={"C1_earlier_operator_is_top_decile_forward_compatible":{"supported":c1},"C2_later_operator_is_not_reverse_compatible":{"supported":c2},"C3_directional_asymmetry_is_large":{"supported":c3,"percentile_gaps":gaps},"C4_both_transplants_beat_deletion":{"supported":c4},"C5_complete_census_is_split_stable":{"supported":c5},"factorization_and_native_invariants":{"supported":maximum_factor_error<=1e-4 and maximum_native_error<=1e-4,"factor_error":maximum_factor_error,"native_error":maximum_native_error}}
    for row in hypotheses.values():row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results,"control_labels":{p["name"]:[labels[c["other"]] for c in p["controls"]] for p in programs}},indent=2,allow_nan=False))
    lines=["# Complete control census for directional operator compatibility","",f"Every learned operator other than the target and partner supplies one exact norm-and-angle-matched control, giving {len(programs[0]['controls'])} controls per direction. Lower percentile is better. `L3H8→L2H5` names the target first: L2H5 is transplanted into L3H8.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Target ← partner / split | Partner KL | Control p10 / median / p90 | Percentile | Delete KL |","|---|---:|---:|---:|---:|"]
    for p in programs:
        for h in halves:
            row=results[f"{p['name']}_{h}"]; lines.append(f"| {p['name']} / {h} | {row['partner_kl']:.3e} | {row['control_p10']:.3e} / {row['control_median']:.3e} / {row['control_p90']:.3e} | {row['partner_percentile']:.3f} | {row['delete_kl']:.3e} |")
    lines += ["","## Guardrails","","The census removes sampling uncertainty from the control bank but reuses the finite evaluation panel from the eight-control result, so it is a post-result red team. All controls match raw Frobenius operator geometry; they do not match downstream Fisher curvature. Directionality can therefore reflect layer-specific destination sensitivity rather than a semantic refinement relation, which is exactly the distinction tested next."]
    REPORT.write_text("\n".join(lines)+"\n");print(json.dumps({"hypotheses":hypotheses,"summary":{k:{x:v for x,v in row.items() if x!='control_kl'} for k,row in results.items()}},indent=2));print(REPORT)


if __name__=="__main__":main()
