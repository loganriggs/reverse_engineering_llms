#!/usr/bin/env python3
"""Patch spectral coordinates from active donors into inactive recipients."""

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
from run_bilin18_spectral_fixed_companions import OUTPUT as CORE_OUTPUT
from run_bilin18_spectral_gate_validation import CONFIG as GATE_CONFIG, OUTPUT as GATE_OUTPUT
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, OUTPUT as PILOT_OUTPUT, SPARSE, TENSOR_ROOT, fisher_fidelity, safe_name
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG=Path("configs/bilin18_spectral_donor_transfer.json")
OUTPUT=TENSOR_ROOT/"spectral_donor_transfer.json"
REPORT=TENSOR_ROOT/"SPECTRAL_DONOR_TRANSFER.md"
FIGURE=TENSOR_ROOT/"spectral_donor_transfer.png"


@torch.inference_mode()
def main():
 config=json.loads(CONFIG.read_text());gate_config=json.loads(GATE_CONFIG.read_text());gate=json.loads(GATE_OUTPUT.read_text());core=json.loads(CORE_OUTPUT.read_text());pilot=json.loads(PILOT_OUTPUT.read_text());arrays=np.load(PILOT_ARRAYS);categories=config["categories"]
 state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True);down=state["transformer.h.1.mlp.Down.weight"].float().cuda();del state
 archive=np.load(SPARSE/"natural_large.npz");train=torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32));mean=train.mean(0).cuda();std=train.std(0).clamp_min(1e-4).cuda();del train,archive
 sae=torch.load(SPARSE/"activation_sae_large_seed101.pt",map_location="cpu",weights_only=True);ew=sae["encoder.weight"].float().cuda();eb=sae["encoder.bias"].float().cuda();decoder_original=std[:,None]*sae["decoder.weight"].float().cuda();del sae
 features={c:pilot["spectra"][f"semantic/seed101/{c}"]["metadata"]["feature"] for c in categories};model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
 all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=gate_config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]]
 def state_at(site):
  prefix=np.array(rows[site["row"],:site["position"]+1],dtype=np.int64,copy=True);ids=torch.as_tensor(prefix,device="cuda")[None];x0,value_bus,pre,normalized=prefix_to_layer1(model,ids);product=model.blocks[1].mlp.Left(normalized)*model.blocks[1].mlp.Right(normalized);return {"x0":x0,"value_bus":value_bus,"pre":pre,"normalized":normalized,"product":product}
 records=[]
 for ci,category in enumerate(categories):
  name=f"semantic/seed101/{category}";feature=features[category];vectors=torch.from_numpy(arrays[f"{safe_name(name+'_raw')}_eigenvectors"][:,:16]).cuda();top_v=vectors[:,:config["rank"]];low_v=vectors[:,16-config["rank"]:16];cross_category=categories[(ci+1)%len(categories)];cross_name=f"semantic/seed101/{cross_category}";cross_v=torch.from_numpy(arrays[f"{safe_name(cross_name+'_raw')}_eigenvectors"][:,:config["rank"]]).cuda()
  donors=[s for s in gate["sites"][category] if s["stratum"]==config["donor_stratum"]];recipients=[s for s in gate["sites"][category] if s["stratum"]==config["recipient_stratum"]]
  for donor_site,recipient_site in zip(donors,recipients):
   donor=state_at(donor_site);recipient=state_at(recipient_site);xd=donor["normalized"][0,-1].float();xr=recipient["normalized"][0,-1].float()
   def raw_patch(v):return ((xd@v)-(xr@v))@v.T
   deltas={"top":raw_patch(top_v),"low":raw_patch(low_v),"cross":raw_patch(cross_v)};cap=config["maximum_input_norm_fraction"]*xr.norm();top_scale=min(1.,float(cap/deltas["top"].norm().clamp_min(1e-20)));deltas["top"]*=top_scale;target_norm=deltas["top"].norm()
   for key in ("low","cross"):deltas[key]*=target_norm/deltas[key].norm().clamp_min(1e-20)
   native_write=model.blocks[1].mlp.Down(recipient["product"])+model.blocks[1].mlp.Down_bias;native_logits=suffix_from_layer1(model,recipient["x0"],recipient["value_bus"],recipient["pre"]+native_write)[0,-1]
   donor_pre=float(F.linear((donor["product"][0,-1].float()-mean)/std,ew,eb)[feature]);recipient_pre=float(F.linear((recipient["product"][0,-1].float()-mean)/std,ew,eb)[feature]);variants={}
   for variant,delta in deltas.items():
    patched=recipient["product"].clone();patched[0,-1]=model.blocks[1].mlp.Left(xr+delta)*model.blocks[1].mlp.Right(xr+delta);patched_write=model.blocks[1].mlp.Down(patched)+model.blocks[1].mlp.Down_bias;patched_logits=suffix_from_layer1(model,recipient["x0"],recipient["value_bus"],recipient["pre"]+patched_write)[0,-1];code_before=sparse_code(recipient["product"][0,-1],mean,std,ew,eb);code_after=sparse_code(patched[0,-1],mean,std,ew,eb);delta_code=code_after-code_before;patched_pre=float(F.linear((patched[0,-1].float()-mean)/std,ew,eb)[feature]);denominator=donor_pre-recipient_pre;transfer=(patched_pre-recipient_pre)/denominator if abs(denominator)>1e-12 else float("nan")
    row={"input_delta_norm":float(delta.norm()),"feature_preactivation":patched_pre,"feature_transfer_fraction":transfer,"target_gate_active":bool(code_after[feature]>0),"downstream_effect_energy":float(fisher_fidelity(patched_logits,patched_logits-native_logits,torch.zeros_like(patched_logits))[2])}
    if variant=="top":
     actual_effect=patched_logits-native_logits
     def mediate(indices):
      indices=torch.as_tensor(indices,device="cuda",dtype=torch.long);recon=decoder_original[:,indices]@delta_code[indices];pred=recipient["product"].clone();pred[0,-1]+=recon;pw=model.blocks[1].mlp.Down(pred)+model.blocks[1].mlp.Down_bias;pl=suffix_from_layer1(model,recipient["x0"],recipient["value_bus"],recipient["pre"]+pw)[0,-1];ff,fc,_=fisher_fidelity(patched_logits,actual_effect,pl-native_logits);return {"fidelity":ff,"cosine":fc}
     row["own_core"]=mediate(core["cores"][category][:config["core_size"]]);row["cross_core"]=mediate(core["cores"][cross_category][:config["core_size"]])
    variants[variant]=row
   records.append({"category":category,"donor":donor_site,"recipient":recipient_site,"donor_feature_preactivation":donor_pre,"recipient_feature_preactivation":recipient_pre,"cap_scale":top_scale,"variants":variants})
 summary={v:{"median_feature_transfer":float(np.median([r["variants"][v]["feature_transfer_fraction"] for r in records])),"gate_activation_rate":float(np.mean([r["variants"][v]["target_gate_active"] for r in records])),"median_effect_energy":float(np.median([r["variants"][v]["downstream_effect_energy"] for r in records]))} for v in ("top","low","cross")};own=float(np.median([r["variants"]["top"]["own_core"]["fidelity"] for r in records]));cross=float(np.median([r["variants"]["top"]["cross_core"]["fidelity"] for r in records]));category_wins={c:float(np.median([r["variants"]["top"]["feature_transfer_fraction"]-r["variants"]["low"]["feature_transfer_fraction"] for r in records if r["category"]==c])) for c in categories}
 hypotheses={"D1_feature_transfer":{"supported":summary["top"]["median_feature_transfer"]>=.15 and summary["top"]["median_feature_transfer"]-summary["low"]["median_feature_transfer"]>=.10,"top":summary["top"]["median_feature_transfer"],"low":summary["low"]["median_feature_transfer"],"gap":summary["top"]["median_feature_transfer"]-summary["low"]["median_feature_transfer"]},"D2_gate_transfer":{"supported":summary["top"]["gate_activation_rate"]>=.25,"activation_rate":summary["top"]["gate_activation_rate"]},"D3_companion_mediation":{"supported":own>=.5,"fidelity":own},"D4_core_specificity":{"supported":own-cross>=.2,"own":own,"cross":cross,"gap":own-cross},"D5_category_consistency":{"supported":sum(v>0 for v in category_wins.values())>=4,"category_gaps":category_wins}}
 output={"config":config,"summary":summary,"records":records,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
 lines=["# Spectral donor-transfer test","","Rank-4 raw spectral coordinates from high-active donors are copied into random-inactive recipients, capped at 10% of recipient input norm. Low-mode and next-category patches have exactly matched edit norm. The frozen 32-feature companion core then predicts the top-patch downstream effect.","","## Locked hypotheses",""]
 for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
 lines += ["","![Donor transfer](spectral_donor_transfer.png)","","| Patch | Feature transfer | Gate activation rate | Effect energy |","|---|---:|---:|---:|"]
 for key,row in summary.items():lines.append(f"| {key} | {row['median_feature_transfer']:.3f} | {row['gate_activation_rate']:.1%} | {row['median_effect_energy']:.3g} |")
 lines += ["",f"Frozen own-core mediation: **{own:.3f}**; next-category core: **{cross:.3f}**.","","Raw donor/recipient pairs are in `spectral_donor_transfer.json`."];REPORT.write_text("\n".join(lines)+"\n")
 fig,axes=plt.subplots(1,2,figsize=(10,4.2));labels=["top","low","cross"];axes[0].bar(labels,[summary[k]["median_feature_transfer"] for k in labels]);axes[0].set_ylabel("median feature transfer fraction");axes[1].bar(["own core","cross core"],[own,cross]);axes[1].set_ylabel("downstream mediation fidelity");fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
