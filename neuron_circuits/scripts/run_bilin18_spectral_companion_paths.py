#!/usr/bin/env python3
"""Decompose residual spectral-path effects into named and companion SAE features."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_gate_validation import CONFIG as GATE_CONFIG, OUTPUT as GATE_OUTPUT
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, SPARSE, TENSOR_ROOT, fisher_fidelity, safe_name
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_companion_paths.json")
OUTPUT = TENSOR_ROOT / "spectral_companion_paths.json"
REPORT = TENSOR_ROOT / "SPECTRAL_COMPANION_PATHS.md"
FIGURE = TENSOR_ROOT / "spectral_companion_paths.png"


def sparse_code(product, mean, std, encoder_weight, encoder_bias):
    pre = F.linear((product.float() - mean) / std, encoder_weight, encoder_bias)
    values = F.relu(pre)
    top = torch.topk(values, 128)
    code = torch.zeros_like(values)
    code.scatter_(0, top.indices, top.values)
    return code


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); gate_config = json.loads(GATE_CONFIG.read_text())
    gate = json.loads(GATE_OUTPUT.read_text()); arrays = np.load(PILOT_ARRAYS)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    down_cpu = state["transformer.h.1.mlp.Down.weight"].float(); del state
    archive = np.load(SPARSE / "natural_large.npz"); train = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)); mean=train.mean(0).cuda();std=train.std(0).clamp_min(1e-4).cuda();del train,archive
    sae=torch.load(SPARSE/"activation_sae_large_seed101.pt",map_location="cpu",weights_only=True)
    encoder_weight=sae["encoder.weight"].float().cuda();encoder_bias=sae["encoder.bias"].float().cuda();decoder=sae["decoder.weight"].float().cuda();del sae
    decoder_original=std[:,None]*decoder
    decoded_write=down_cpu.cuda()@decoder_original
    feature_write_norm=decoded_write.square().sum(0).sqrt()
    pilot=json.load(open(TENSOR_ROOT/"spectral_spike_pilot.json"))
    feature_ids={category:pilot["spectra"][f"semantic/seed101/{category}"]["metadata"]["feature"] for category in config["categories"]}
    feature_names={value:key for key,value in feature_ids.items()}
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=gate_config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]]
    records=[]
    for category in config["categories"]:
        name=f"semantic/seed101/{category}";key=safe_name(f"{name}_{config['basis']}")
        vectors=torch.from_numpy(arrays[f"{key}_eigenvectors"][:,:config["rank"]]).cuda();feature=feature_ids[category]
        source=[site for site in gate["sites"][category] if site["stratum"]==config["source_stratum"]]
        for site in source:
            prefix=np.array(rows[site["row"],:site["position"]+1],dtype=np.int64,copy=True);ids=torch.as_tensor(prefix,device="cuda")[None]
            x0,value_bus,pre,normalized=prefix_to_layer1(model,ids);product=model.blocks[1].mlp.Left(normalized)*model.blocks[1].mlp.Right(normalized)
            native_write=model.blocks[1].mlp.Down(product)+model.blocks[1].mlp.Down_bias;native_logits=suffix_from_layer1(model,x0,value_bus,pre+native_write)[0,-1]
            x=normalized[0,-1].float();delta=-(x@vectors)@vectors.T;delta*=config["input_norm_fraction"]*x.norm()/delta.norm().clamp_min(1e-20);modified_x=x+delta
            modified_product=product.clone();modified_product[0,-1]=model.blocks[1].mlp.Left(modified_x)*model.blocks[1].mlp.Right(modified_x)
            actual_write=model.blocks[1].mlp.Down(modified_product)+model.blocks[1].mlp.Down_bias;actual_logits=suffix_from_layer1(model,x0,value_bus,pre+actual_write)[0,-1]
            actual_effect=actual_logits-native_logits;direct_target=actual_write[0,-1]-native_write[0,-1]
            before=sparse_code(product[0,-1],mean,std,encoder_weight,encoder_bias);after=sparse_code(modified_product[0,-1],mean,std,encoder_weight,encoder_bias);delta_code=after-before
            contribution=delta_code.abs()*feature_write_norm;order=torch.argsort(contribution,descending=True);nonzero=int((delta_code!=0).sum())
            target_rank=int((order==feature).nonzero()[0])+1 if delta_code[feature]!=0 else None
            energies=contribution[order].square();cumulative=energies.cumsum(0)/energies.sum().clamp_min(1e-30);needed90=int(torch.searchsorted(cumulative,.9).item()+1)

            def evaluate_indices(indices):
                reconstructed=decoder_original[:,indices]@delta_code[indices]
                predicted_product=product.clone();predicted_product[0,-1]+=reconstructed
                predicted_write=model.blocks[1].mlp.Down(predicted_product)+model.blocks[1].mlp.Down_bias
                predicted_logits=suffix_from_layer1(model,x0,value_bus,pre+predicted_write)[0,-1]
                downstream,cosine,_=fisher_fidelity(actual_logits,actual_effect,predicted_logits-native_logits)
                direct,direct_cosine,_=euclidean_fidelity(direct_target,predicted_write[0,-1]-native_write[0,-1])
                return {"downstream_fidelity":downstream,"downstream_cosine":cosine,"direct_fidelity":direct,"direct_cosine":direct_cosine}

            target_metrics=evaluate_indices(torch.as_tensor([feature],device="cuda"))
            budgets={}
            for budget in config["budgets"]:
                indices=order[:min(budget,nonzero)]
                budgets[str(budget)]={**evaluate_indices(indices),"features":indices.cpu().tolist(),"known_feature_names":[feature_names.get(int(i)) for i in indices.cpu().tolist()]}
            full_indices=torch.nonzero(delta_code).flatten();full_metrics=evaluate_indices(full_indices)
            records.append({"category":category,**site,"nonzero_code_changes":nonzero,"target_feature":feature,"target_feature_rank":target_rank,"features_for_90pct_individual_write_energy":needed90,"target_only":target_metrics,"budgets":budgets,"full_sae":full_metrics,"top_companions":[{"feature":int(i),"known_name":feature_names.get(int(i)),"delta_code":float(delta_code[i]),"write_score":float(contribution[i])} for i in order[:16]]})
    summaries={}
    for key in ["target_only"]+[str(b) for b in config["budgets"]]+["full_sae"]:
        rows_for_key=[row[key] if key in ("target_only","full_sae") else row["budgets"][key] for row in records]
        summaries[key]={metric:float(np.median([row[metric] for row in rows_for_key])) for metric in ("downstream_fidelity","direct_fidelity")}
    target_top16=float(np.mean([row["target_feature_rank"] is not None and row["target_feature_rank"]<=16 for row in records]));needed90=float(np.median([row["features_for_90pct_individual_write_energy"] for row in records]))
    full=summaries["full_sae"]["downstream_fidelity"];rank64=summaries["64"]["downstream_fidelity"];target=summaries["target_only"]["downstream_fidelity"]
    hypotheses={
      "C1_full_sae_delta":{"supported":summaries["full_sae"]["direct_fidelity"]>=.5 and full>=.5,**summaries["full_sae"]},
      "C2_companion_gain":{"supported":rank64-target>=.2,"rank64":rank64,"target_only":target,"gain":rank64-target},
      "C3_companion_compactness":{"supported":rank64>=.8*full,"rank64":rank64,"full_sae":full,"fraction":rank64/max(full,1e-30)},
      "C4_target_prominence":{"supported":target_top16>=.5,"fraction_top16":target_top16},
      "C5_sparse_write_energy":{"supported":needed90<=64,"median_features":needed90}}
    output={"config":config,"summaries":summaries,"target_top16_fraction":target_top16,"median_features_for_90pct":needed90,"records":records,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=["# SAE companion-path decomposition","","For each known high-active spectral edit, this decomposes the actual native-product change into changes of the same SAE's Top-128 code. Companion features are ranked by `|Δcode| × decoded downstream-write norm` and reinserted before the native Down projection.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Companion paths](spectral_companion_paths.png)","","| Reconstruction | Direct-write fidelity | Downstream fidelity |","|---|---:|---:|"]
    for key,row in summaries.items():lines.append(f"| {key} | {row['direct_fidelity']:.3f} | {row['downstream_fidelity']:.3f} |")
    lines += ["",f"The named feature is top-16 by changed decoded-write contribution on **{target_top16:.1%}** of sites. The median number of changed features needed for 90% of summed individual decoded-write energy is **{needed90:.0f}**.","","Raw feature identities and per-site reconstructions are in `spectral_companion_paths.json`."];REPORT.write_text("\n".join(lines)+"\n")
    keys=["target_only"]+[str(b) for b in config["budgets"]]+["full_sae"];x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(x,[summaries[k]["direct_fidelity"] for k in keys],marker="o",label="direct write");ax.plot(x,[summaries[k]["downstream_fidelity"] for k in keys],marker="o",label="downstream");ax.set_xticks(x,keys,rotation=25,ha="right");ax.set_ylabel("median fidelity");ax.legend();fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
