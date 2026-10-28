#!/usr/bin/env python3
"""Transfer frozen SAE companion cores to independent random-active semantic pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_matched_semantics import CONFIG as MATCH_CONFIG, OUTPUT as MATCH_OUTPUT
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, SPARSE, TENSOR_ROOT, fisher_fidelity, safe_name
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG=Path("configs/bilin18_spectral_companion_transfer.json")
OUTPUT=TENSOR_ROOT/"spectral_companion_transfer.json"
REPORT=TENSOR_ROOT/"SPECTRAL_COMPANION_TRANSFER.md"
FIGURE=TENSOR_ROOT/"spectral_companion_transfer.png"


@torch.inference_mode()
def main():
 config=json.loads(CONFIG.read_text());core=json.loads(CORE_OUTPUT.read_text());matched=json.loads(MATCH_OUTPUT.read_text());match_config=json.loads(MATCH_CONFIG.read_text());arrays=np.load(PILOT_ARRAYS);categories=core["config"]["categories"]
 state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True);down=state["transformer.h.1.mlp.Down.weight"].float().cuda();del state
 archive=np.load(SPARSE/"natural_large.npz");train=torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32));mean=train.mean(0).cuda();std=train.std(0).clamp_min(1e-4).cuda();del train,archive
 sae=torch.load(SPARSE/"activation_sae_large_seed101.pt",map_location="cpu",weights_only=True);ew=sae["encoder.weight"].float().cuda();eb=sae["encoder.bias"].float().cuda();decoder=sae["decoder.weight"].float().cuda();del sae
 decoder_original=std[:,None]*decoder;feature_write_norm=(down@decoder_original).square().sum(0).sqrt();model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
 all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=match_config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]];records=[]
 for ci,category in enumerate(categories):
  name=f"semantic/seed101/{category}";vectors=torch.from_numpy(arrays[f"{safe_name(name+'_raw')}_eigenvectors"][:,:4]).cuda();cross=categories[(ci+1)%len(categories)]
  for stratum in config["source_strata"]:
   sites=[s for s in matched["sites"][category] if s["stratum"]==stratum]
   for site in sites:
    prefix=np.array(rows[site["row"],:site["position"]+1],dtype=np.int64,copy=True);ids=torch.as_tensor(prefix,device="cuda")[None];x0,value_bus,pre,normalized=prefix_to_layer1(model,ids);product=model.blocks[1].mlp.Left(normalized)*model.blocks[1].mlp.Right(normalized)
    native_write=model.blocks[1].mlp.Down(product)+model.blocks[1].mlp.Down_bias;native_logits=suffix_from_layer1(model,x0,value_bus,pre+native_write)[0,-1];x=normalized[0,-1].float();delta=-(x@vectors)@vectors.T;delta*=.1*x.norm()/delta.norm().clamp_min(1e-20);modified=product.clone();modified[0,-1]=model.blocks[1].mlp.Left(x+delta)*model.blocks[1].mlp.Right(x+delta)
    actual_write=model.blocks[1].mlp.Down(modified)+model.blocks[1].mlp.Down_bias;actual_logits=suffix_from_layer1(model,x0,value_bus,pre+actual_write)[0,-1];actual_effect=actual_logits-native_logits;direct_target=actual_write[0,-1]-native_write[0,-1]
    delta_code=sparse_code(modified[0,-1],mean,std,ew,eb)-sparse_code(product[0,-1],mean,std,ew,eb);contribution=delta_code.abs()*feature_write_norm;adaptive=torch.argsort(contribution,descending=True);nonzero=int((delta_code!=0).sum())
    def evaluate(indices):
     indices=torch.as_tensor(indices,device="cuda",dtype=torch.long);recon=decoder_original[:,indices]@delta_code[indices];pred=product.clone();pred[0,-1]+=recon;pw=model.blocks[1].mlp.Down(pred)+model.blocks[1].mlp.Down_bias;pl=suffix_from_layer1(model,x0,value_bus,pre+pw)[0,-1];df,dc,_=euclidean_fidelity(direct_target,pw[0,-1]-native_write[0,-1]);ff,fc,_=fisher_fidelity(actual_logits,actual_effect,pl-native_logits);return {"downstream_fidelity":ff,"downstream_cosine":fc,"direct_fidelity":df,"direct_cosine":dc}
    budgets={}
    for budget in config["budgets"]:budgets[str(budget)]={"fixed":evaluate(core["cores"][category][:budget]),"adaptive":evaluate(adaptive[:min(budget,nonzero)]),"cross_category":evaluate(core["cores"][cross][:budget])}
    records.append({"category":category,"stratum":stratum,"pair_id":site["pair_id"],"budgets":budgets})
 summaries={}
 for budget in config["budgets"]:
  summaries[str(budget)]={}
  for stratum in config["source_strata"]:
   subset=[r for r in records if r["stratum"]==stratum];summaries[str(budget)][stratum]={v:{m:float(np.median([r["budgets"][str(budget)][v][m] for r in subset])) for m in ("downstream_fidelity","direct_fidelity")} for v in ("fixed","adaptive","cross_category")}
 b16=summaries["16"];pos=b16["random_positive"]["fixed"]["downstream_fidelity"];neg=b16["random_negative"]["fixed"]["downstream_fidelity"];adaptive=b16["random_positive"]["adaptive"]["downstream_fidelity"]
 category_gaps=[]
 for c in categories:
  p=np.median([r["budgets"]["16"]["fixed"]["downstream_fidelity"] for r in records if r["category"]==c and r["stratum"]=="random_positive"]);n=np.median([r["budgets"]["16"]["fixed"]["downstream_fidelity"] for r in records if r["category"]==c and r["stratum"]=="random_negative"]);category_gaps.append(float(p-n))
 hypotheses={"T1_random_positive_transfer":{"supported":pos>=.5,"fidelity":pos},"T2_semantic_specificity":{"supported":pos-neg>=.2,"positive":pos,"negative":neg,"gap":pos-neg},"T3_adaptive_retention":{"supported":pos>=.8*adaptive,"fixed":pos,"adaptive":adaptive,"fraction":pos/max(adaptive,1e-30)},"T4_small_core_transfer":{"supported":summaries["8"]["random_positive"]["fixed"]["downstream_fidelity"]>=.4,"fidelity":summaries["8"]["random_positive"]["fixed"]["downstream_fidelity"]},"T5_category_consistency":{"supported":all(x>0 for x in category_gaps),"category_gaps":dict(zip(categories,category_gaps))}}
 output={"config":config,"records":records,"summaries":summaries,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
 lines=["# Frozen companion-core transfer","","Cores discovered on even-indexed high-active sites are applied unchanged to a separate FineWeb panel of random-active surface positives and activation-matched surface negatives.","","## Locked hypotheses",""]
 for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
 lines += ["","![Companion transfer](spectral_companion_transfer.png)","","| Budget | Positive fixed | Positive adaptive | Positive cross-category | Matched-negative fixed |","|---:|---:|---:|---:|---:|"]
 for budget,row in summaries.items():lines.append(f"| {budget} | {row['random_positive']['fixed']['downstream_fidelity']:.3f} | {row['random_positive']['adaptive']['downstream_fidelity']:.3f} | {row['random_positive']['cross_category']['downstream_fidelity']:.3f} | {row['random_negative']['fixed']['downstream_fidelity']:.3f} |")
 lines += ["", "Per-category paired results are in `spectral_companion_transfer.json`."];REPORT.write_text("\n".join(lines)+"\n")
 xs=config["budgets"];fig,ax=plt.subplots(figsize=(8,4.5));
 for label,stratum,variant in (("positive fixed","random_positive","fixed"),("positive adaptive","random_positive","adaptive"),("matched-negative fixed","random_negative","fixed")):ax.plot(xs,[summaries[str(b)][stratum][variant]["downstream_fidelity"] for b in xs],marker="o",label=label)
 ax.set_xlabel("SAE features");ax.set_ylabel("median downstream fidelity");ax.legend();fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
