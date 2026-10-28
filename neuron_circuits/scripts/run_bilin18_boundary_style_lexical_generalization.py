#!/usr/bin/env python3
"""Prospectively test the two-hop style circuit on new spelling lexemes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_two_hop_circuit import collect
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_boundary_style_lexical_generalization.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_lexical_generalization.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_LEXICAL_GENERALIZATION.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_lexical_generalization.png"


@torch.inference_mode()
def native_logits(model, ids):
    dimension=model.config.n_embd; x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; bus=None
    for block in model.blocks: x,bus=block(x,bus,x0)
    return 30*torch.tanh(model.lm_head(F.rms_norm(x,(dimension,))[:,-1]).float()/30)


@torch.inference_mode()
def global_swap(model, raw, records, contrasts, batch_size):
    declared=[]; random=[]
    for start in range(0,len(records),batch_size):
        batch=records[start:start+batch_size]; original=np.stack([raw[r[0],:128].copy() for r in batch]); swapped=original.copy()
        for i,r in enumerate(batch): swapped[i,r[3]]=r[5]
        logits=native_logits(model,torch.as_tensor(np.concatenate([original,swapped]),device="cuda")).reshape(2,len(batch),-1)
        change=(logits[1]-logits[0])@contrasts; direction=torch.as_tensor([1 if not r[4] else -1 for r in batch],device="cuda",dtype=torch.float32)
        declared.append((direction*change[:,0]).cpu()); random.append((direction[:,None]*change[:,1:]).cpu())
    declared=torch.cat(declared); random=torch.cat(random).mean(0); mean=float(declared.mean())
    return {"win":float((declared>0).float().mean()),"mean":mean,"random_percentile":float((random<mean).float().mean())}


def token_pairs(tokenizer, pairs):
    result=[]
    for us,uk in pairs:
        a=tokenizer.encode(us,add_special_tokens=False); b=tokenizer.encode(uk,add_special_tokens=False)
        if len(a)!=1 or len(b)!=1: raise ValueError(f"pair is not single-token: {us}/{uk}")
        result.append((a[0],b[0]))
    return result


def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size)
    flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool)
    old_pairs=token_pairs(tokenizer,base["spelling_pairs"]); new_pairs=token_pairs(tokenizer,config["new_spelling_pairs"])
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    cases=(("new_cues_old_targets",new_pairs,old_pairs),("old_cues_new_targets",old_pairs,new_pairs)); results=[]
    for case,cues,targets in cases:
        contrasts=make_contrasts(model.lm_head.weight.shape[0],targets,config["random_contrasts"],config["seed"]+(0 if case.startswith("new") else 100),"cuda")
        for split,(begin,end) in enumerate(base["splits"]):
            raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary,cues,counts,127,config["examples_per_label_per_split"],config["seed"]+split+(0 if case.startswith("new") else 100))
            print("lexical",case,split,len(records),flush=True)
            global_result=global_swap(model,raw,records,contrasts,config["batch_size"])
            programs,error=collect(model,raw,records,contrasts,counts,boundary,config)
            results.append({"case":case,"split":split,"examples":len(records),"global":global_result,"native_error":error,"programs":programs})
    predicted="L8H2_cue__L9H8_boundary"
    g0=all(r["global"]["win"]>=.75 for r in results); g1=all(r["programs"][predicted]["native_win"]>=.65 for r in results); g2=all(r["programs"][predicted]["mediation_fraction"]>=.40 for r in results)
    specificity=[]
    for r in results:
        p=r["programs"][predicted]; upstream=all(p["native_mean"]>=5*max(abs(r["programs"][f"{u}__L9H8_boundary"]["native_mean"]),1e-30) for u in ("L8H1_cue","L8H2_noncue")); downstream=all(p["mediation_fraction"]>=r["programs"][f"L8H2_cue__{d}"]["mediation_fraction"]+.20 for d in ("L9H7_boundary","L9H8_nonboundary")); specificity.append(upstream and downstream)
    g3=all(specificity); g4=all(r["programs"][predicted]["native_random_percentile"]>=.95 and r["programs"][predicted]["mediated_random_percentile"]>=.95 for r in results)
    hypotheses={"G0_native_model_generalizes_both_directions":{"supported":g0},"G1_L8H2_cue_edge_generalizes_both_directions":{"supported":g1},"G2_L9H8_mediation_generalizes":{"supported":g2},"G3_route_specificity_generalizes":{"supported":g3},"G4_declared_contrasts_beat_random":{"supported":g4},"G5_typed_circuit_generalizes_new_lexemes":{"supported":all((g0,g1,g2,g3,g4))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Lexical generalization of the boundary-style circuit","","Eleven single-token US/UK spelling pairs that were absent from circuit discovery are used in both directions: as new cue swaps with the original targets, and as new output targets with the original cues. The native model, finite L8H2 cue-edge intervention, L9H8 boundary-edge freeze, upstream/downstream controls, and random signed contrasts are all reevaluated.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Case/split | N | model win/mean | L8 edge win/mean | L9-mediated fraction | random percentiles |","|---|---:|---:|---:|---:|---:|"]
    for r in results:
        p=r["programs"][predicted]; lines.append(f"| {r['case']}/{r['split']} | {r['examples']} | {r['global']['win']:.3f}/{r['global']['mean']:+.4g} | {p['native_win']:.3f}/{p['native_mean']:+.4g} | {p['mediation_fraction']:.3f} | {p['native_random_percentile']:.3f}/{p['mediated_random_percentile']:.3f} |")
    lines += ["","## Interpretation boundary","","This is lexical transfer within the same manually declared US/UK spelling family, not transfer to an unrelated style variable. The new list was fixed after checking only single-token tokenization and corpus availability, before inspecting model or circuit effects. Passing supports a reusable typed lexical-style circuit rather than a twelve-pair lookup-table explanation."]
    REPORT.write_text("\n".join(lines)+"\n")
    x=np.arange(len(results)); labels=[f"{r['case'][:3]}s{r['split']}" for r in results]; fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained"); ax.bar(x-.2,[r["global"]["win"] for r in results],.4,label="whole model"); ax.bar(x+.2,[r["programs"][predicted]["native_win"] for r in results],.4,label="L8H2 cue edge"); ax.axhline(.5,color="black",linewidth=.8); ax.set(xticks=x,xticklabels=labels,ylabel="Correct shift fraction",title="Generalization to eleven new spelling pairs"); ax.legend(); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"results":results},indent=2)); print(REPORT)


if __name__=="__main__": main()
