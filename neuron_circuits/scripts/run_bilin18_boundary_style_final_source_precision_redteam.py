#!/usr/bin/env python3
"""Float64 closure red team for the terminal spelling-source factorial."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts,select_records
from run_bilin18_boundary_style_final_source_factorial import FACTORS,collect
from run_bilin18_layer01_supervised_feature_census import FEATURES,vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT

CONFIG=Path("configs/bilin18_boundary_style_final_source_precision_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_final_source_precision_redteam.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_FINAL_SOURCE_PRECISION_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_final_source_precision_redteam.png"

def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); prior=json.loads(Path(config["source_result"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool); pairs=[(tokenizer.encode(a,add_special_tokens=False)[0],tokenizer.encode(b,add_special_tokens=False)[0]) for a,b in base["spelling_pairs"]]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results=[]
    for fold in range(config["folds"]):
        targets=set(range(fold*config["target_pairs_per_fold"],(fold+1)*config["target_pairs_per_fold"])); target_pairs=[p for i,p in enumerate(pairs) if i in targets]; cues=[p for i,p in enumerate(pairs) if i not in targets]; contrasts=make_contrasts(model.lm_head.weight.shape[0],target_pairs,config["random_contrasts"],config["seed"]+fold,"cuda")
        for split,(begin,end) in enumerate(base["splits"]):
            raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary,cues,counts,127,config["examples_per_label_per_split"],config["seed"]+fold*10+split); print("precision",fold,split,len(records),flush=True); results.append({"fold":fold,"split":split,**collect(model,raw,records,contrasts,config)})
    lookup={(r["fold"],r["split"]):r for r in prior["results"]}; max_state=max(r["state_recomposition_error"] for r in results); max_output=max(r["output_recomposition_error"] for r in results); max_edge=max(r["edge_recomposition_error"] for r in results); n0=max_state<=1e-10 and max_output<=1e-6 and max_edge<=5e-4
    differences=[]
    for r in results:
        old=lookup[(r["fold"],r["split"])]
        for name in FACTORS: differences.append(abs(r["sources"][name]["projection_share"]-old["sources"][name]["projection_share"]))
    n1=max(differences)<=.02; n2=all(r["sources"]["carried"]["projection_share"]>=.30 and r["sources"]["attention_combined"]["projection_share"]>=.20 for r in results); n3=all(r["sources"]["nonboundary_attention"]["mean"]>r["sources"]["boundary_attention"]["mean"] for r in results); n4=all(abs(r["sources"]["mlp"]["projection_share"])<.05 for r in results)
    hypotheses={"N0_double_source_closure_is_exact":{"supported":n0,"max_state":max_state,"max_output":max_output,"raw_edge_discrepancy":max_edge},"N1_source_shares_are_precision_stable":{"supported":n1,"max_change":max(differences)},"N2_carried_and_attention_conclusion_survives":{"supported":n2},"N3_nonboundary_attention_conclusion_survives":{"supported":n3},"N4_final_MLP_is_negligible":{"supported":n4},"N5_terminal_two_branch_ontology_is_supported":{"supported":all((n0,n1,n2,n3,n4))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Precision red team of the final spelling-source factorial","","The initial float32 audit failed its 1e-5 structural closure gate despite exact finite output Shapley closure. This rerun accumulates all residual and edge components in float64, defines nonboundary attention as the exact residual after the boundary-edge component, and compares every behavioral share with the original run.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"Maximum float64 state closure: **{max_state:.3g}**. Maximum output closure: **{max_output:.3g}**. Raw native-attention/edge-sum discrepancy: **{max_edge:.3g}**. Maximum share change: **{max(differences):.3g}**.","","| Fold/split | carried | boundary attention | nonboundary attention | MLP |","|---|---:|---:|---:|---:|"]
    for r in results: lines.append(f"| {r['fold']}/{r['split']} | {r['sources']['carried']['projection_share']:.3f} | {r['sources']['boundary_attention']['projection_share']:.3f} | {r['sources']['nonboundary_attention']['projection_share']:.3f} | {r['sources']['mlp']['projection_share']:.3f} |")
    lines += ["","The raw edge-sum discrepancy is retained as a numerical audit of reproducing the checkpoint's float32 contraction order; it is not silently relabeled exact. Float64 residual closure tests the algebraic source partition, while exact finite output closure tests the behavioral allocation."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained"); x=np.arange(len(results)); labels=[f"f{r['fold']}s{r['split']}" for r in results]
    for name in FACTORS: ax.plot(x,[r["sources"][name]["projection_share"] for r in results],marker="o",label=name)
    ax.set(xticks=x,xticklabels=labels,ylabel="Projection share",title="Float64 terminal-source replication"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses},indent=2)); print(REPORT)

if __name__=="__main__": main()
