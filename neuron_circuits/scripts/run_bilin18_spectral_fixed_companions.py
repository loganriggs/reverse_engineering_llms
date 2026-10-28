#!/usr/bin/env python3
"""Freeze category-specific SAE companion cores and validate them on held-out sites."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_gate_validation import CONFIG as GATE_CONFIG, OUTPUT as GATE_OUTPUT
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, SPARSE, TENSOR_ROOT, fisher_fidelity, safe_name
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_fixed_companions.json")
OUTPUT = TENSOR_ROOT / "spectral_fixed_companions.json"
REPORT = TENSOR_ROOT / "SPECTRAL_FIXED_COMPANIONS.md"
FIGURE = TENSOR_ROOT / "spectral_fixed_companions.png"


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text());gate_config=json.loads(GATE_CONFIG.read_text());gate=json.loads(GATE_OUTPUT.read_text());arrays=np.load(PILOT_ARRAYS)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True);down=state["transformer.h.1.mlp.Down.weight"].float().cuda();del state
    archive=np.load(SPARSE/"natural_large.npz");train=torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32));mean=train.mean(0).cuda();std=train.std(0).clamp_min(1e-4).cuda();del train,archive
    sae=torch.load(SPARSE/"activation_sae_large_seed101.pt",map_location="cpu",weights_only=True);ew=sae["encoder.weight"].float().cuda();eb=sae["encoder.bias"].float().cuda();decoder=sae["decoder.weight"].float().cuda();del sae
    decoder_original=std[:,None]*decoder;feature_write_norm=(down@decoder_original).square().sum(0).sqrt()
    pilot=json.load(open(TENSOR_ROOT/"spectral_spike_pilot.json"));feature_ids={c:pilot["spectra"][f"semantic/seed101/{c}"]["metadata"]["feature"] for c in config["categories"]}
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=gate_config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]]

    def compute(category,site,need_logits):
        name=f"semantic/seed101/{category}";vectors=torch.from_numpy(arrays[f"{safe_name(name+'_raw')}_eigenvectors"][:,:4]).cuda()
        prefix=np.array(rows[site["row"],:site["position"]+1],dtype=np.int64,copy=True);ids=torch.as_tensor(prefix,device="cuda")[None]
        x0,value_bus,pre,normalized=prefix_to_layer1(model,ids);product=model.blocks[1].mlp.Left(normalized)*model.blocks[1].mlp.Right(normalized)
        x=normalized[0,-1].float();delta=-(x@vectors)@vectors.T;delta*=.1*x.norm()/delta.norm().clamp_min(1e-20);modified=product.clone();modified[0,-1]=model.blocks[1].mlp.Left(x+delta)*model.blocks[1].mlp.Right(x+delta)
        before=sparse_code(product[0,-1],mean,std,ew,eb);after=sparse_code(modified[0,-1],mean,std,ew,eb);delta_code=after-before
        result={"delta_code":delta_code,"contribution":delta_code.abs()*feature_write_norm,"product":product,"modified":modified,"x0":x0,"value_bus":value_bus,"pre":pre}
        if need_logits:
            native_write=model.blocks[1].mlp.Down(product)+model.blocks[1].mlp.Down_bias;actual_write=model.blocks[1].mlp.Down(modified)+model.blocks[1].mlp.Down_bias
            native_logits=suffix_from_layer1(model,x0,value_bus,pre+native_write)[0,-1];actual_logits=suffix_from_layer1(model,x0,value_bus,pre+actual_write)[0,-1]
            result.update(native_write=native_write,actual_write=actual_write,native_logits=native_logits,actual_logits=actual_logits)
        return result

    cores={};discovery={}
    for category in config["categories"]:
        source=[s for s in gate["sites"][category] if s["stratum"]=="high_active"]
        disc=source[::2];scores=[]
        for site in disc:scores.append(compute(category,site,False)["contribution"].cpu())
        mean_score=torch.stack(scores).mean(0);order=torch.argsort(mean_score,descending=True);cores[category]=order[:max(config["budgets"])].tolist()
        discovery[category]={"sites":disc,"top_features":cores[category],"named_feature":feature_ids[category],"named_rank":int((order==feature_ids[category]).nonzero()[0])+1}

    records=[]
    for ci,category in enumerate(config["categories"]):
        cross_category=config["categories"][(ci+1)%len(config["categories"])]
        source=[s for s in gate["sites"][category] if s["stratum"]=="high_active"][1::2]
        for site in source:
            data=compute(category,site,True);delta_code=data["delta_code"];adaptive=torch.argsort(data["contribution"],descending=True)
            actual_effect=data["actual_logits"]-data["native_logits"];direct_target=data["actual_write"][0,-1]-data["native_write"][0,-1]
            def eval_features(indices):
                indices=torch.as_tensor(indices,device="cuda",dtype=torch.long);reconstruction=decoder_original[:,indices]@delta_code[indices];pred_product=data["product"].clone();pred_product[0,-1]+=reconstruction
                pred_write=model.blocks[1].mlp.Down(pred_product)+model.blocks[1].mlp.Down_bias;pred_logits=suffix_from_layer1(model,data["x0"],data["value_bus"],data["pre"]+pred_write)[0,-1]
                downstream,cosine,_=fisher_fidelity(data["actual_logits"],actual_effect,pred_logits-data["native_logits"]);direct,dc,_=euclidean_fidelity(direct_target,pred_write[0,-1]-data["native_write"][0,-1]);return {"downstream_fidelity":downstream,"downstream_cosine":cosine,"direct_fidelity":direct,"direct_cosine":dc}
            budgets={}
            nonzero=int((delta_code!=0).sum())
            for budget in config["budgets"]:
                budgets[str(budget)]={"fixed":eval_features(cores[category][:budget]),"adaptive":eval_features(adaptive[:min(budget,nonzero)]),"cross_category":eval_features(cores[cross_category][:budget])}
            records.append({"category":category,"cross_category":cross_category,**site,"budgets":budgets})
    summaries={}
    for budget in config["budgets"]:
        summaries[str(budget)]={variant:{metric:float(np.median([r["budgets"][str(budget)][variant][metric] for r in records])) for metric in ("downstream_fidelity","direct_fidelity")} for variant in ("fixed","adaptive","cross_category")}
    fixed16=summaries["16"]["fixed"]["downstream_fidelity"];adaptive16=summaries["16"]["adaptive"]["downstream_fidelity"];cross16=summaries["16"]["cross_category"]["downstream_fidelity"]
    hypotheses={
      "F1_fixed_core_fidelity":{"supported":fixed16>=.5,"fidelity":fixed16},
      "F2_adaptive_retention":{"supported":fixed16>=.8*adaptive16,"fixed":fixed16,"adaptive":adaptive16,"fraction":fixed16/max(adaptive16,1e-30)},
      "F3_category_specificity":{"supported":fixed16-cross16>=.2,"fixed":fixed16,"cross_category":cross16,"gap":fixed16-cross16},
      "F4_named_feature_core":{"supported":all(feature_ids[c] in cores[c][:16] for c in config["categories"]),"named_ranks":{c:discovery[c]["named_rank"] for c in config["categories"]}},
      "F5_small_fixed_core":{"supported":summaries["8"]["fixed"]["downstream_fidelity"]>=.4,"fidelity":summaries["8"]["fixed"]["downstream_fidelity"]}}
    output={"config":config,"cores":cores,"discovery":discovery,"records":records,"summaries":summaries,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=["# Frozen SAE companion-core validation","","Category-specific companion features are ranked by mean changed decoded-write magnitude on even-indexed high-active sites, frozen, and evaluated on odd-indexed sites. A next-category core is the specificity control.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Fixed companions](spectral_fixed_companions.png)","","| Budget | Fixed core | Adaptive per-site | Cross-category control | Fixed direct |","|---:|---:|---:|---:|---:|"]
    for budget,row in summaries.items():lines.append(f"| {budget} | {row['fixed']['downstream_fidelity']:.3f} | {row['adaptive']['downstream_fidelity']:.3f} | {row['cross_category']['downstream_fidelity']:.3f} | {row['fixed']['direct_fidelity']:.3f} |")
    lines += ["", "Core memberships, named-feature ranks, and validation records are in `spectral_fixed_companions.json`."];REPORT.write_text("\n".join(lines)+"\n")
    xs=config["budgets"];fig,ax=plt.subplots(figsize=(8,4.5))
    for variant in ("fixed","adaptive","cross_category"):ax.plot(xs,[summaries[str(b)][variant]["downstream_fidelity"] for b in xs],marker="o",label=variant)
    ax.set_xlabel("SAE features");ax.set_ylabel("median downstream fidelity");ax.legend();fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
