#!/usr/bin/env python3
"""Scale sweep for directional compatibility of complete value-side operators."""

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


CONFIG=Path("configs/bilin18_full_operator_scale_sweep.json")
CENSUS=TENSOR_ROOT/"bilin18_full_operator_control_census.json"
OUTPUT=TENSOR_ROOT/"bilin18_full_operator_scale_sweep.json"
REPORT=TENSOR_ROOT/"BILIN18_FULL_OPERATOR_SCALE_SWEEP.md"
PARTIAL=TENSOR_ROOT/"bilin18_full_operator_scale_sweep.partial.json"


def main():
    config=json.loads(CONFIG.read_text());model=load_full_model("cuda")
    model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    bank,gram,norms=build_function_bank(model,[0,1,2,3,4,5]);labels=[row["label"] for row in bank]
    pair_labels=[f"L{l}H{h}" for l,h in config["pair"]];pair=[labels.index(x) for x in pair_labels]
    programs=[]
    for d in range(2):
        target,partner=pair[d],pair[1-d];scale,controls,cosine=complete_controls(gram,norms,target,partner)
        programs.append({"name":f"{pair_labels[d]}_to_{pair_labels[1-d]}","target":target,"partner":partner,"partner_scale":scale,"controls":controls,"cosine":cosine})
    corpus=np.load(config["corpus"],mmap_mode="r");raw=np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"]+config["evaluation_rows"],:config["sequence_length"]+1],dtype=np.int64)
    halves={"a":raw[:len(raw)//2],"b":raw[len(raw)//2:]};positions=torch.as_tensor(config["positions"],device="cuda")
    saved=json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {};results=saved.get("results",{}) if saved.get("config")==config else {}
    maximum_factor_error=float(saved.get("maximum_factor_error",0)) if saved.get("config")==config else 0.0
    for intervention_scale in config["scales"]:
        for program in programs:
            for half,panel in halves.items():
                key=f"{program['name']}_{half}_{intervention_scale}"
                if key in results:print("scale sweep",key,"(resumed)",flush=True);continue
                print("scale sweep",key,flush=True);totals=torch.zeros(3+len(program["controls"]),dtype=torch.float64,device="cuda");sites=0
                for start in range(0,len(panel),config["evaluation_batch_size"]):
                    ids=torch.as_tensor(panel[start:start+config["evaluation_batch_size"],:config["sequence_length"]],device="cuda")
                    logits,error=intervention_logits(model,ids,bank[program["target"]],bank[program["partner"]],program["partner_scale"],program["controls"],bank,positions,intervention_scale)
                    maximum_factor_error=max(maximum_factor_error,error);logp=F.log_softmax(logits[0].float(),-1)
                    for variant in range(1,len(logits)):
                        logq=F.log_softmax(logits[variant].float(),-1);totals[variant]+=(logp.exp()*(logp-logq)).sum(-1).clamp_min(0).double().sum()
                    sites+=logits.shape[1]*logits.shape[2]
                means=(totals/sites).cpu().numpy();controls=means[3:]
                results[key]={"partner_kl":float(means[2]),"control_median":float(np.median(controls)),"partner_percentile":float(np.mean(controls<=means[2])),"partner_control_ratio":float(means[2]/max(np.median(controls),1e-30))}
                PARTIAL.write_text(json.dumps({"config":config,"results":results,"maximum_factor_error":maximum_factor_error},indent=2))
    census=json.loads(CENSUS.read_text())["results"]
    for program in programs:
        for half in halves:
            source=census[f"{program['name']}_{half}"];results[f"{program['name']}_{half}_1.0"]={key:source[key] for key in ("partner_kl","control_median","partner_percentile","partner_control_ratio")}
    forward=next(p for p in programs if p["name"].startswith("L3H8"));reverse=next(p for p in programs if p["name"].startswith("L2H5"));all_scales=config["scales"]+[1.0]
    def row(program,half,scale):return results[f"{program['name']}_{half}_{scale}"]
    s1=all(row(forward,h,s)["partner_percentile"]<=.10 for h in halves for s in all_scales)
    ranges={h:max(row(forward,h,s)["partner_percentile"] for s in all_scales)-min(row(forward,h,s)["partner_percentile"] for s in all_scales) for h in halves};s2=all(x<=.15 for x in ranges.values())
    s3=all(row(reverse,h,s)["partner_percentile"]>.10 for h in halves for s in all_scales)
    gaps={f"{h}_{s}":row(reverse,h,s)["partner_percentile"]-row(forward,h,s)["partner_percentile"] for h in halves for s in all_scales};s4=all(x>=.30 for x in gaps.values())
    quadratic={}
    for p in programs:
        for h in halves:
            low=row(p,h,.05)["partner_kl"]/.05**2;high=row(p,h,.2)["partner_kl"]/.2**2;quadratic[f"{p['name']}_{h}"]=max(low,high)/max(min(low,high),1e-30)
    s5=all(x<=2 for x in quadratic.values())
    hypotheses={"S1_forward_compatibility_is_local":{"supported":s1},"S2_forward_percentile_is_scale_stable":{"supported":s2,"ranges":ranges},"S3_reverse_direction_never_enters_top_decile":{"supported":s3},"S4_directional_gap_persists_locally":{"supported":s4,"gaps":gaps},"S5_small_scale_effect_is_approximately_quadratic":{"supported":s5,"ratios":quadratic},"factorization_invariant":{"supported":maximum_factor_error<=1e-4,"maximum_error":maximum_factor_error}}
    for value in hypotheses.values():value["supported"]=bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False));PARTIAL.unlink(missing_ok=True)
    lines=["# Scale sweep for directional full-operator compatibility","","The complete 52-control census is repeated at small and intermediate interpolation scales. This distinguishes a local destination-metric relation from a large finite-edit accident. Scale 1.0 is copied from the complete census on the same panel; smaller scales use its first 64 rows.","","## Locked hypotheses",""]
    for name,value in hypotheses.items():lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["","| Target ← partner / half | Scale | Partner percentile | Partner/control | Partner KL |","|---|---:|---:|---:|---:|"]
    for p in programs:
        for h in halves:
            for s in all_scales:
                value=row(p,h,s);lines.append(f"| {p['name']} / {h} | {s:.2f} | {value['partner_percentile']:.3f} | {value['partner_control_ratio']:.3f} | {value['partner_kl']:.3e} |")
    lines += ["","## Guardrails","","The smaller-scale panels are subsets of the scale-1 census rather than new data. At very small KL, float precision and logit saturation can affect ranks; approximate quadratic scaling is therefore an explicit invariant. Persistence across scale identifies local metric compatibility, not semantic equivalence or a transferable QK router."]
    REPORT.write_text("\n".join(lines)+"\n");print(json.dumps({"hypotheses":hypotheses},indent=2));print(REPORT)


if __name__=="__main__":main()
