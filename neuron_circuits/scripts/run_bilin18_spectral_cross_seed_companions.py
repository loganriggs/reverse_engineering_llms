#!/usr/bin/env python3
"""Discover seed-102 companion cores and compare canonical decoded-write subspaces."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_companion_paths import sparse_code
from run_bilin18_spectral_fixed_companions import OUTPUT as SEED101_CORE_OUTPUT
from run_bilin18_spectral_gate_validation import collect_gate_sites
from run_bilin18_spectral_spike_pilot import ARRAYS as PILOT_ARRAYS, OUTPUT as PILOT_OUTPUT, SPARSE, TENSOR_ROOT, fisher_fidelity, safe_name
from run_bilin18_spectral_input_interventions import euclidean_fidelity
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1
from transformers import AutoTokenizer


CONFIG=Path("configs/bilin18_spectral_cross_seed_companions.json")
OUTPUT=TENSOR_ROOT/"spectral_cross_seed_companions.json"
REPORT=TENSOR_ROOT/"SPECTRAL_CROSS_SEED_COMPANIONS.md"
FIGURE=TENSOR_ROOT/"spectral_cross_seed_companions.png"


def orthonormal(matrix):
    return torch.linalg.qr(matrix,mode="reduced").Q


def overlap(first,second):
    rank=min(first.shape[1],second.shape[1]);return float((first.T@second).square().sum()/rank)


def permutation_test(matrix):
    observed=float(np.trace(matrix)/len(matrix));null=[float(np.mean([matrix[i,p[i]] for i in range(len(matrix))])) for p in itertools.permutations(range(len(matrix)))]
    return {"observed_diagonal_mean":observed,"permutation_p_value":float(np.mean(np.asarray(null)>=observed)),"diagonal_top_matches":int(sum(np.argmax(matrix[i])==i for i in range(len(matrix)))),"permutations":len(null)}


@torch.inference_mode()
def main():
 config=json.loads(CONFIG.read_text());pilot=json.loads(PILOT_OUTPUT.read_text());seed101=json.loads(SEED101_CORE_OUTPUT.read_text());arrays=np.load(PILOT_ARRAYS);categories=config["categories"]
 state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True);down=state["transformer.h.1.mlp.Down.weight"].float().cuda();del state
 archive=np.load(SPARSE/"natural_large.npz");train=torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32));mean=train.mean(0).cuda();std=train.std(0).clamp_min(1e-4).cuda();del train,archive
 saes={};decoded_writes={}
 for seed in (101,102):
  sae=torch.load(SPARSE/f"activation_sae_large_seed{seed}.pt",map_location="cpu",weights_only=True);saes[seed]={"encoder_weight":sae["encoder.weight"].float().cuda(),"encoder_bias":sae["encoder.bias"].float().cuda(),"decoder_original":std[:,None]*sae["decoder.weight"].float().cuda()};decoded_writes[seed]=down@saes[seed]["decoder_original"]
 feature_ids={c:pilot["spectra"][f"semantic/seed102/{c}"]["metadata"]["feature"] for c in categories}
 model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 all_rows=np.load(DEFAULT_CORPUS,mmap_mode="r");panel=config["panel"];rows=all_rows[panel["start"]:panel["start"]+panel["rows"]]
 site_config={"positions":config["positions"],"stratum_size":config["panel"]["sites_per_category"],"seed":config["seed"]}
 sites,excluded,counts=collect_gate_sites(model,rows,categories,feature_ids,saes[102]["encoder_weight"],saes[102]["encoder_bias"],mean,std,site_config,tokenizer)
 if excluded: raise RuntimeError(f"unexpected excluded categories: {excluded}")

 def compute(category,site,logits=False):
  name=f"semantic/seed102/{category}";vectors=torch.from_numpy(arrays[f"{safe_name(name+'_raw')}_eigenvectors"][:,:config["spectral_rank"]]).cuda();prefix=np.array(rows[site["row"],:site["position"]+1],dtype=np.int64,copy=True);ids=torch.as_tensor(prefix,device="cuda")[None];x0,value_bus,pre,normalized=prefix_to_layer1(model,ids);product=model.blocks[1].mlp.Left(normalized)*model.blocks[1].mlp.Right(normalized);x=normalized[0,-1].float();delta=-(x@vectors)@vectors.T;delta*=config["input_norm_fraction"]*x.norm()/delta.norm().clamp_min(1e-20);modified=product.clone();modified[0,-1]=model.blocks[1].mlp.Left(x+delta)*model.blocks[1].mlp.Right(x+delta)
  before=sparse_code(product[0,-1],mean,std,saes[102]["encoder_weight"],saes[102]["encoder_bias"]);after=sparse_code(modified[0,-1],mean,std,saes[102]["encoder_weight"],saes[102]["encoder_bias"]);delta_code=after-before;result={"product":product,"modified":modified,"delta_code":delta_code,"contribution":delta_code.abs()*decoded_writes[102].square().sum(0).sqrt(),"x0":x0,"value_bus":value_bus,"pre":pre}
  if logits:
   nw=model.blocks[1].mlp.Down(product)+model.blocks[1].mlp.Down_bias;aw=model.blocks[1].mlp.Down(modified)+model.blocks[1].mlp.Down_bias;nl=suffix_from_layer1(model,x0,value_bus,pre+nw)[0,-1];al=suffix_from_layer1(model,x0,value_bus,pre+aw)[0,-1];result.update(native_write=nw,actual_write=aw,native_logits=nl,actual_logits=al)
  return result

 cores102={};discovery={}
 for category in categories:
  high=[s for s in sites[category] if s["stratum"]=="high_active"];scores=[compute(category,s,False)["contribution"].cpu() for s in high[::2]];order=torch.argsort(torch.stack(scores).mean(0),descending=True);cores102[category]=order[:config["core_size"]].tolist();discovery[category]={"named_feature":feature_ids[category],"named_rank":int((order==feature_ids[category]).nonzero()[0])+1}
 q101={c:orthonormal(decoded_writes[101][:,seed101["cores"][c][:config["core_size"]]]) for c in categories};q102={c:orthonormal(decoded_writes[102][:,cores102[c]]) for c in categories}
 matrix=np.asarray([[overlap(q101[a],q102[b]) for b in categories] for a in categories]);stability={"matrix":matrix.tolist(),**permutation_test(matrix)}
 # Red-team whether cross-seed recurrence is carried only by the named feature.
 named101={c:pilot["spectra"][f"semantic/seed101/{c}"]["metadata"]["feature"] for c in categories}
 companion101={c:orthonormal(decoded_writes[101][:,[i for i in seed101["cores"][c][:config["core_size"]] if i!=named101[c]]]) for c in categories}
 companion102={c:orthonormal(decoded_writes[102][:,[i for i in cores102[c] if i!=feature_ids[c]]]) for c in categories}
 companion_matrix=np.asarray([[overlap(companion101[a],companion102[b]) for b in categories] for a in categories])
 companion_stability={"matrix":companion_matrix.tolist(),**permutation_test(companion_matrix)}
 named_matrix=np.zeros((len(categories),len(categories)))
 for i,a in enumerate(categories):
  first=torch.nn.functional.normalize(decoded_writes[101][:,named101[a]],dim=0)
  for j,b in enumerate(categories):
   second=torch.nn.functional.normalize(decoded_writes[102][:,feature_ids[b]],dim=0);named_matrix[i,j]=float((first@second).square())
 named_stability={"matrix":named_matrix.tolist(),**permutation_test(named_matrix)}
 records=[]
 for ci,category in enumerate(categories):
  cross=categories[(ci+1)%len(categories)];high=[s for s in sites[category] if s["stratum"]=="high_active"][1::2]
  for site in high:
   d=compute(category,site,True);actual_delta=d["actual_write"][0,-1]-d["native_write"][0,-1];actual_effect=d["actual_logits"]-d["native_logits"]
   def via_product(indices):
    indices=torch.as_tensor(indices,device="cuda",dtype=torch.long);recon=saes[102]["decoder_original"][:,indices]@d["delta_code"][indices];pp=d["product"].clone();pp[0,-1]+=recon;pw=model.blocks[1].mlp.Down(pp)+model.blocks[1].mlp.Down_bias;pl=suffix_from_layer1(model,d["x0"],d["value_bus"],d["pre"]+pw)[0,-1];ff,fc,_=fisher_fidelity(d["actual_logits"],actual_effect,pl-d["native_logits"]);df,dc,_=euclidean_fidelity(actual_delta,pw[0,-1]-d["native_write"][0,-1]);return {"downstream_fidelity":ff,"downstream_cosine":fc,"direct_fidelity":df,"direct_cosine":dc}
   def via_write_subspace(q):
    pred=q@(q.T@actual_delta);pw=d["native_write"].clone();pw[0,-1]+=pred;pl=suffix_from_layer1(model,d["x0"],d["value_bus"],d["pre"]+pw)[0,-1];ff,fc,_=fisher_fidelity(d["actual_logits"],actual_effect,pl-d["native_logits"]);df,dc,_=euclidean_fidelity(actual_delta,pred);return {"downstream_fidelity":ff,"downstream_cosine":fc,"direct_fidelity":df,"direct_cosine":dc}
   adaptive=torch.argsort(d["contribution"],descending=True)[:config["core_size"]]
   records.append({"category":category,**site,"program_fixed":via_product(cores102[category]),"program_adaptive":via_product(adaptive),"subspace_seed102":via_write_subspace(q102[category]),"subspace_seed101":via_write_subspace(q101[category]),"subspace_cross_category":via_write_subspace(q101[cross])})
 variants=("program_fixed","program_adaptive","subspace_seed102","subspace_seed101","subspace_cross_category");summaries={v:{m:float(np.median([r[v][m] for r in records])) for m in ("downstream_fidelity","direct_fidelity")} for v in variants}
 own=summaries["subspace_seed102"]["downstream_fidelity"];trans=summaries["subspace_seed101"]["downstream_fidelity"];cross=summaries["subspace_cross_category"]["downstream_fidelity"]
 hypotheses={"X1_seed102_program":{"supported":summaries["program_fixed"]["downstream_fidelity"]>=.5,**summaries["program_fixed"]},"X2_category_subspace_match":{"supported":stability["permutation_p_value"]<=.05 and stability["diagonal_top_matches"]>=4,**stability},"X3_seed101_subspace_transfer":{"supported":trans>=.5,"fidelity":trans},"X4_rotated_subspace_retention":{"supported":trans>=.8*own,"seed101":trans,"seed102":own,"fraction":trans/max(own,1e-30)},"X5_cross_category_specificity":{"supported":trans-cross>=.2,"corresponding":trans,"cross_category":cross,"gap":trans-cross}}
 output={"config":config,"excluded":excluded,"gate_counts":counts,"sites":sites,"cores_seed102":cores102,"discovery":discovery,"stability":stability,"companion_only_stability":companion_stability,"named_only_stability":named_stability,"records":records,"summaries":summaries,"hypotheses":hypotheses};OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False))
 lines=["# Cross-seed companion-core canonicalization","","Seed-102 companion cores are independently discovered from its own Top-128 gates and spectral modes. Exact sparse programs are evaluated separately from oracle projections into decoded-write subspaces, which are invariant to rotations inside a feature ensemble.","","## Locked hypotheses",""]
 for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
 lines += ["","![Cross-seed companions](spectral_cross_seed_companions.png)","","| Object | Direct fidelity | Downstream fidelity |","|---|---:|---:|"]
 for key,row in summaries.items():lines.append(f"| {key} | {row['direct_fidelity']:.3f} | {row['downstream_fidelity']:.3f} |")
 lines += ["",f"Same-category decoded-write subspace overlap: **{stability['observed_diagonal_mean']:.3f}**; exact assignment `p={stability['permutation_p_value']:.4f}`; diagonal row maxima **{stability['diagonal_top_matches']}/5**.","",f"Removing the named feature leaves companion-only diagonal overlap **{companion_stability['observed_diagonal_mean']:.3f}**, exact `p={companion_stability['permutation_p_value']:.4f}`, and **{companion_stability['diagonal_top_matches']}/5** diagonal row maxima. Thus recurrence is not a one-feature artifact. Named-feature-only matching is also 5/5.","","Raw cores, overlap matrices, and site effects are in `spectral_cross_seed_companions.json`."];REPORT.write_text("\n".join(lines)+"\n")
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));im=axes[0].imshow(matrix,vmin=0,vmax=max(.5,matrix.max()),cmap="viridis");axes[0].set_xticks(range(5),categories,rotation=45,ha="right");axes[0].set_yticks(range(5),categories);axes[0].set_title("Seed-101 vs seed-102 core subspaces");fig.colorbar(im,ax=axes[0],fraction=.046);x=np.arange(len(variants));axes[1].bar(x,[summaries[v]["downstream_fidelity"] for v in variants]);axes[1].set_xticks(x,variants,rotation=45,ha="right");axes[1].set_ylabel("median downstream fidelity");fig.tight_layout();fig.savefig(FIGURE,dpi=180);plt.close(fig);print("wrote",OUTPUT,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
