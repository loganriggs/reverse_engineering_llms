#!/usr/bin/env python3
"""Replicate prospectively selected downstream spelling consumers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_downstream_screen import collect
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_boundary_style_downstream_replication.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_downstream_replication.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_DOWNSTREAM_REPLICATION.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_downstream_replication.png"


def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool)
    pairs=[(tokenizer.encode(a,add_special_tokens=False)[0],tokenizer.encode(b,add_special_tokens=False)[0]) for a,b in base["spelling_pairs"]]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results=[]
    for fold in range(3):
        targets=set(range(fold*config["target_pairs_per_fold"],(fold+1)*config["target_pairs_per_fold"])); target_pairs=[p for i,p in enumerate(pairs) if i in targets]; cues=[p for i,p in enumerate(pairs) if i not in targets]
        contrasts=make_contrasts(model.lm_head.weight.shape[0],target_pairs,config["random_contrasts"],config["seed"]+fold,"cuda")
        for split,(begin,end) in enumerate(base["splits"]):
            if fold==0 and split==0: continue
            raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary,cues,counts,127,config["examples_per_label"],config["seed"]+fold*10+split)
            print("replicate downstream",fold,split,len(records),flush=True); rows,error=collect(model,raw,records,contrasts,config); results.append({"fold":fold,"split":split,"native_error":error,"programs":rows})
    r0=all(r["native_error"]<=1e-4 for r in results)
    l17=[r["programs"]["L17_h2"] for r in results]; r1cells=sum(x["mediation_fraction"]>=.05 and x["random_percentile"]>=.95 for x in l17); r1=r1cells>=4
    full=[r["programs"]["L17_full_destination"] for r in results]; r2cells=sum(x["mediation_fraction"]>=.20 for x in full); r2=r2cells>=4
    r3=all(r["programs"]["L11_h6"]["mediation_fraction"]<.05 for r in results)
    r4cells=sum(r["programs"]["L17_full_destination"]["mediation_fraction"]>=r["programs"]["L17_all_boundary"]["mediation_fraction"]+.10 for r in results); r4=r4cells>=4
    hypotheses={"R0_native_endpoints_are_consistent":{"supported":r0},"R1_L17H2_boundary_replicates":{"supported":r1,"cells":r1cells},"R2_L17_full_destination_replicates":{"supported":r2,"cells":r2cells},"R3_L11H6_is_weak_not_major":{"supported":r3},"R4_boundary_only_is_incomplete_at_L17":{"supported":r4,"cells":r4cells},"R5_residual_consumer_is_distributed":{"supported":all((r1,r2,r3,r4))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Held-out replication of downstream spelling consumers","","L17H2, complete Layer-17 destination attention, and L11H6 were selected from the fold-0/split-0 screen. The complete 81-edge screen is rerun on the other five fold/split cells, but only these locked claims are evaluated.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Fold/split | L9H8 | L17H2 | L17 all-boundary | L17 full-destination | L11H6 |","|---|---:|---:|---:|---:|---:|"]
    for r in results: lines.append(f"| {r['fold']}/{r['split']} | {r['programs']['L9_h8']['mediation_fraction']:.3f} | {r['programs']['L17_h2']['mediation_fraction']:.3f} | {r['programs']['L17_all_boundary']['mediation_fraction']:.3f} | {r['programs']['L17_full_destination']['mediation_fraction']:.3f} | {r['programs']['L11_h6']['mediation_fraction']:.3f} |")
    lines += ["","## Interpretation boundary","","Replication of complete destination attention without comparable boundary-edge mediation means the suffix consumes the spelling edit through a broader contextual attention update. It does not by itself distinguish many individually tiny causal edges from an interaction or a changed destination query. The final-layer tensor factors and additive residual branches are the next localization boundary."]
    REPORT.write_text("\n".join(lines)+"\n")
    x=np.arange(len(results)); labels=[f"f{r['fold']}s{r['split']}" for r in results]; fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for name,label in (("L9_h8","L9H8 boundary"),("L17_h2","L17H2 boundary"),("L17_all_boundary","L17 all boundary"),("L17_full_destination","L17 full destination")): ax.plot(x,[r["programs"][name]["mediation_fraction"] for r in results],marker="o",label=label)
    ax.set(xticks=x,xticklabels=labels,ylabel="Mediated fraction",title="Held-out downstream consumers"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"summary":[{"fold":r["fold"],"split":r["split"],"L17H2":r["programs"]["L17_h2"]["mediation_fraction"],"L17full":r["programs"]["L17_full_destination"]["mediation_fraction"]} for r in results]},indent=2)); print(REPORT)


if __name__=="__main__": main()
