#!/usr/bin/env python3
"""Decode a held-out semantic profile for the dominant destination-Fisher mode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_destination_fisher_geometry import cosine_gram
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank, intervention_logits
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG=Path("configs/bilin18_destination_fisher_mode_semantics.json")
FISHER=TENSOR_ROOT/"bilin18_destination_fisher_geometry.pt"
OUTPUT=TENSOR_ROOT/"bilin18_destination_fisher_mode_semantics.json"
REPORT=TENSOR_ROOT/"BILIN18_DESTINATION_FISHER_MODE_SEMANTICS.md"


def clean(tokenizer,index):return tokenizer.decode([int(index)]).replace("\n","\\n").replace("\t","\\t")


def main():
    config=json.loads(CONFIG.read_text());saved=torch.load(FISHER,map_location="cpu",weights_only=True);fisher_a=saved["fisher_a"].numpy();values,vectors=np.linalg.eigh(fisher_a);mode=torch.as_tensor(vectors[:,-1],device="cuda",dtype=torch.float32)
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);flags=np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]);categories=token_categories(flags)
    model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    bank,gram,norms=build_function_bank(model,[0,1,2,3,4,5]);labels=[row["label"] for row in bank];target_label=f"L{config['target'][0]}H{config['target'][1]}";partner_label=f"L{config['partner'][0]}H{config['partner'][1]}";target=labels.index(target_label);partner=labels.index(partner_label);partner_scale,controls,_=complete_controls(gram,norms,target,partner)
    corpus=np.load(config["corpus"],mmap_mode="r")
    if "evaluation_random_seed" in config:
        generator=np.random.default_rng(config["evaluation_random_seed"]);indices=np.sort(generator.choice(config["evaluation_index_high"],config["evaluation_rows"],replace=False));raw=np.asarray(corpus[indices,:config["sequence_length"]+1],dtype=np.int64)
    else:
        indices=np.arange(config["evaluation_start"],config["evaluation_start"]+config["evaluation_rows"]);raw=np.asarray(corpus[indices,:config["sequence_length"]+1],dtype=np.int64)
    positions=torch.as_tensor(config["positions"],device="cuda")
    site_energy=[];site_targets=[];site_context=[];effect_sum=torch.zeros(model.config.vocab_size,dtype=torch.float64,device="cuda");effect_square=torch.zeros_like(effect_sum);fisher_gram=torch.zeros(len(mode),len(mode),dtype=torch.float64,device="cuda")
    for row_index in range(len(raw)):
        if row_index%8==0:print("decode Fisher mode rows",row_index,"/",len(raw),flush=True)
        ids=torch.as_tensor(raw[row_index:row_index+1,:config["sequence_length"]],device="cuda")
        logits,_=intervention_logits(model,ids,bank[target],bank[partner],partner_scale,controls,bank,positions,config["intervention_scale"])
        native=logits[0].float();delta=(logits[2:].float()-native[None])/config["intervention_scale"];probability=F.softmax(native,-1);mean=(delta*probability[None]).sum(-1,keepdim=True);centered=delta-mean;weighted=centered*probability.sqrt()[None];flat=weighted.flatten(1).double();fisher_gram+=flat@flat.T
        mode_effect=torch.einsum("c,cbpv->bpv",mode,centered);mode_mean=(mode_effect*probability).sum(-1,keepdim=True);mode_centered=mode_effect-mode_mean;energy=(probability*mode_centered.square()).sum(-1)
        effect_sum+=mode_centered.double().sum((0,1));effect_square+=mode_centered.double().square().sum((0,1))
        for local,position in enumerate(config["positions"]):
            site_energy.append(float(energy[0,local]));site_targets.append(int(raw[row_index,position+1]));left=max(0,position-24);text=tokenizer.decode(raw[row_index,left:position+1].tolist()).replace("\n","\\n")
            site_context.append({"row":int(indices[row_index]),"position":position,"target":clean(tokenizer,raw[row_index,position+1]),"context":text})
    fisher_gram=(fisher_gram/(len(raw)*len(config["positions"]))).cpu().numpy();capture=float(mode.cpu().numpy()@fisher_gram@mode.cpu().numpy()/max(np.trace(fisher_gram),1e-30))
    energies=np.asarray(site_energy);targets=np.asarray(site_targets);feature_rows=[]
    for index,name in enumerate(FEATURES):
        positive=flags[targets,index].astype(bool);count=int(positive.sum())
        if count<config["minimum_feature_sites"] or count==len(positive):continue
        ratio=float(energies[positive].mean()/max(energies[~positive].mean(),1e-30));feature_rows.append({"feature":name,"sites":count,"energy_ratio":ratio,"positive_mean":float(energies[positive].mean()),"negative_mean":float(energies[~positive].mean())})
    feature_rows.sort(key=lambda row:row["energy_ratio"],reverse=True)
    mean_effect=(effect_sum/len(energies)).cpu().numpy()[:tokenizer.vocab_size];rms_effect=(effect_square/len(energies)).sqrt().cpu().numpy()[:tokenizer.vocab_size]
    if abs(mean_effect.min())>mean_effect.max():mean_effect=-mean_effect
    positive_ids=np.argsort(mean_effect)[::-1][:config["top_tokens"]];negative_ids=np.argsort(mean_effect)[:config["top_tokens"]];rms_ids=np.argsort(rms_effect)[::-1][:config["top_tokens"]]
    def pole(ids):
        counts=np.bincount(categories[ids],minlength=len(CATEGORIES));return {"tokens":[clean(tokenizer,index) for index in ids],"purity":float(counts.max()/len(ids)),"plurality":CATEGORIES[int(counts.argmax())]}
    poles={"positive":pole(positive_ids),"negative":pole(negative_ids),"rms":pole(rms_ids)}
    order=np.argsort(energies)[::-1];contexts=[]
    for index in order[:config["top_contexts"]]:contexts.append({**site_context[int(index)],"energy":float(energies[index]),"share":float(energies[index]/max(energies.sum(),1e-30))})
    maximum_share=float(energies.max()/max(energies.sum(),1e-30));best_ratio=feature_rows[0]["energy_ratio"] if feature_rows else 0;max_purity=max(poles["positive"]["purity"],poles["negative"]["purity"])
    hypotheses={"M1_dominant_mode_transfers_to_fresh_text":{"supported":capture>=.40,"energy_fraction":capture},"M2_mode_energy_has_a_known_semantic_condition":{"supported":best_ratio>=2,"best":feature_rows[0] if feature_rows else None},"M3_signed_output_poles_are_lexically_coherent":{"supported":max_purity>=.75,"positive_purity":poles["positive"]["purity"],"negative_purity":poles["negative"]["purity"]},"M4_mode_is_not_only_one_outlier_context":{"supported":maximum_share<.10,"maximum_site_share":maximum_share}}
    for value in hypotheses.values():value["supported"]=bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"feature_enrichment":feature_rows,"output_poles":poles,"top_contexts":contexts},indent=2,allow_nan=False))
    lines=["# Held-out semantics of the dominant destination-Fisher mode","",f"The leading candidate-space sensitivity mode is frozen from the earlier L3H8 Fisher panel, then applied to {len(energies)} positions from later FineWeb rows. Energy is next-token Fisher energy of the combined logit tangent; no semantic label is used to define the mode.","","## Locked hypotheses",""]
    for name,value in hypotheses.items():lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["",f"Fresh-panel Fisher energy captured: `{capture:.3f}`. Largest single-site share: `{maximum_share:.3%}`.","","Top known next-token enrichments:",""]
    for row in feature_rows[:12]:lines.append(f"- `{row['feature']}`: {row['sites']} sites, energy ratio `{row['energy_ratio']:.2f}`")
    lines += ["",f"Positive pole ({poles['positive']['plurality']}, purity {poles['positive']['purity']:.2f}): `{poles['positive']['tokens']}`",f"Negative pole ({poles['negative']['plurality']}, purity {poles['negative']['purity']:.2f}): `{poles['negative']['tokens']}`",f"Largest-RMS tokens: `{poles['rms']['tokens']}`","","Highest-energy contexts:",""]
    for row in contexts[:16]:lines.append(f"- `{row['context']}` → `{row['target']}`; energy `{row['energy']:.3e}`")
    lines += ["","## Guardrails","","This is a semantic audit of a destination-sensitivity mode, not a native neuron. Feature ratios are exploratory after the mode was frozen and are not multiple-comparison corrected. Mean logit poles can be affected by vocabulary frequency and unembedding anisotropy. A readable label requires replication or an intervention selected from these examples; unexplained but stable remains an acceptable outcome."]
    REPORT.write_text("\n".join(lines)+"\n");print(json.dumps({"hypotheses":hypotheses,"top_features":feature_rows[:5],"poles":poles},indent=2));print(REPORT)


if __name__=="__main__":main()
