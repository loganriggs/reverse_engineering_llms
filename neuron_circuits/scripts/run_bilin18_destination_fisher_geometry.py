#!/usr/bin/env python3
"""Construct the destination-conditioned Fisher geometry of operator replacements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_bus_aware_function_templates import rank_summary
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank, intervention_logits
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import spearman


CONFIG=Path("configs/bilin18_destination_fisher_geometry.json")
ATLAS=TENSOR_ROOT/"bilin18_destination_compatibility_atlas.json"
OUTPUT=TENSOR_ROOT/"bilin18_destination_fisher_geometry.json"
TENSORS=TENSOR_ROOT/"bilin18_destination_fisher_geometry.pt"
REPORT=TENSOR_ROOT/"BILIN18_DESTINATION_FISHER_GEOMETRY.md"


def cosine_gram(gram):
    diagonal=np.sqrt(np.clip(np.diag(gram),1e-30,None));return gram/(diagonal[:,None]*diagonal[None,:])


def main():
    config=json.loads(CONFIG.read_text());model=load_full_model("cuda")
    model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    bank,raw_gram,norms=build_function_bank(model,[0,1,2,3,4,5]);labels=[row["label"] for row in bank]
    target_label=f"L{config['target'][0]}H{config['target'][1]}";partner_label=f"L{config['partner'][0]}H{config['partner'][1]}";target=labels.index(target_label);partner=labels.index(partner_label)
    partner_scale,controls,_=complete_controls(raw_gram,norms,target,partner);candidate_labels=[partner_label]+[labels[row["other"]] for row in controls]
    coefficients=[]
    partner_coefficient=np.zeros(len(bank));partner_coefficient[target]=-1;partner_coefficient[partner]=partner_scale;coefficients.append(partner_coefficient)
    for row in controls:
        coefficient=np.zeros(len(bank));coefficient[target]=row["target_coefficient"]-1;coefficient[row["other"]]=row["other_coefficient"];coefficients.append(coefficient)
    coefficients=np.stack(coefficients);raw_delta_gram=coefficients@raw_gram@coefficients.T;raw_cosine=cosine_gram(raw_delta_gram)
    corpus=np.load(config["corpus"],mmap_mode="r")
    if "evaluation_random_seed" in config:
        generator=np.random.default_rng(config["evaluation_random_seed"]);indices=np.sort(generator.choice(config["evaluation_index_high"],config["evaluation_rows"],replace=False));raw=np.asarray(corpus[indices,:config["sequence_length"]+1],dtype=np.int64)
    else:
        raw=np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"]+config["evaluation_rows"],:config["sequence_length"]+1],dtype=np.int64)
    halves={"a":raw[:len(raw)//2],"b":raw[len(raw)//2:]};positions=torch.as_tensor(config["positions"],device="cuda");fisher={};finite_kl={}
    for half,panel in halves.items():
        print("destination Fisher",half,flush=True);gram=torch.zeros(len(candidate_labels),len(candidate_labels),dtype=torch.float64,device="cuda");kl=torch.zeros(len(candidate_labels),dtype=torch.float64,device="cuda");sites=0
        for start in range(0,len(panel),config["evaluation_batch_size"]):
            ids=torch.as_tensor(panel[start:start+config["evaluation_batch_size"],:config["sequence_length"]],device="cuda")
            logits,_=intervention_logits(model,ids,bank[target],bank[partner],partner_scale,controls,bank,positions,config["intervention_scale"])
            native=logits[0].float();candidate=logits[2:].float();delta=(candidate-native[None])/config["intervention_scale"]
            probability=F.softmax(native,-1);mean=(delta*probability[None]).sum(-1,keepdim=True);centered=delta-mean
            weighted=centered*probability.sqrt()[None];flat=weighted.flatten(1).double();gram+=flat@flat.T
            logp=F.log_softmax(native,-1)
            for index in range(len(candidate_labels)):
                logq=F.log_softmax(candidate[index],-1);kl[index]+=(logp.exp()*(logp-logq)).sum(-1).clamp_min(0).double().sum()
            sites+=native.shape[0]*native.shape[1]
        fisher[half]=(gram/sites).cpu().numpy();finite_kl[half]=(kl/sites).cpu().numpy()
    cosine={half:cosine_gram(value) for half,value in fisher.items()};triangle=np.triu_indices(len(candidate_labels),1);cosine_stability=float(np.corrcoef(cosine["a"][triangle],cosine["b"][triangle])[0,1])
    summaries={half:rank_summary(value) for half,value in fisher.items()};rank=config["fisher_subspace_rank"]
    eigenspaces=[]; dominant={}
    for half in halves:
        values,vectors=np.linalg.eigh(fisher[half]);eigenspaces.append(vectors[:,-rank:])
        leading=vectors[:,-1]; order=np.argsort(np.abs(leading))[::-1]
        dominant[half]={"eigenvalue":float(values[-1]),"partner_abs_loading":float(abs(leading[0])),"partner_loading_rank":int(np.where(order==0)[0][0]+1),"top_loadings":[{"candidate":candidate_labels[int(index)],"loading":float(leading[index])} for index in order[:12]]}
    canonical=np.linalg.svd(eigenspaces[0].T@eigenspaces[1],compute_uv=False);subspace_stability=float(np.square(canonical).mean())
    raw_correlations={half:spearman(raw_cosine[triangle],cosine[half][triangle]) for half in halves};diagonal_correlations={half:spearman(np.diag(fisher[half]),finite_kl[half]) for half in halves}
    hypotheses={
        "F1_destination_fisher_geometry_is_low_rank":{"supported":all(summaries[h]["rank90"]<=16 for h in halves),"rank90":{h:summaries[h]["rank90"] for h in halves}},
        "F2_fisher_cosine_geometry_is_split_stable":{"supported":cosine_stability>=.80,"correlation":cosine_stability},
        "F3_leading_fisher_subspace_is_split_stable":{"supported":subspace_stability>=.70,"mean_squared_canonical_cosine":subspace_stability,"canonical_cosines":canonical.tolist()},
        "F4_raw_operator_geometry_predicts_destination_geometry":{"supported":all(value>=.30 for value in raw_correlations.values()),"correlations":raw_correlations},
        "F5_fisher_diagonal_predicts_small_finite_kl":{"supported":all(value>=.95 for value in diagonal_correlations.values()),"correlations":diagonal_correlations},
    }
    for value in hypotheses.values():value["supported"]=bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"rank_summaries":summaries,"dominant_modes":dominant,"candidate_labels":candidate_labels,"raw_correlations":raw_correlations,"finite_kl":{h:dict(zip(candidate_labels,finite_kl[h].tolist())) for h in halves}},indent=2,allow_nan=False))
    torch.save({"candidate_labels":candidate_labels,"raw_cosine":torch.from_numpy(raw_cosine),"fisher_a":torch.from_numpy(fisher["a"]),"fisher_b":torch.from_numpy(fisher["b"]),"cosine_a":torch.from_numpy(cosine["a"]),"cosine_b":torch.from_numpy(cosine["b"])},TENSORS)
    lines=["# Destination-conditioned Fisher geometry of operator replacements","",f"Fifty-three exact norm-and-angle-matched replacements for {target_label} are propagated at scale {config['intervention_scale']}. Their logit tangents define a Fisher Gram matrix under the native next-token distribution. This quotient asks which operator differences the fixed router, destination state, and nonlinear suffix can see.","","## Locked hypotheses",""]
    for name,value in hypotheses.items():lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["",f"Fisher rank90 A/B: `{summaries['a']['rank90']}/{summaries['b']['rank90']}`; participation ratio `{summaries['a']['participation_ratio']:.1f}/{summaries['b']['participation_ratio']:.1f}`.",f"Cosine-geometry split correlation: `{cosine_stability:.3f}`. Leading-{rank} subspace mean squared canonical cosine: `{subspace_stability:.3f}`.",f"Raw-operator/Fisher cosine Spearman A/B: `{raw_correlations['a']:.3f}/{raw_correlations['b']:.3f}`.",f"Fisher diagonal/finite-KL Spearman A/B: `{diagonal_correlations['a']:.3f}/{diagonal_correlations['b']:.3f}`.","",f"The designated {partner_label} replacement has absolute loading `{dominant['a']['partner_abs_loading']:.3f}/{dominant['b']['partner_abs_loading']:.3f}` on the dominant sensitive mode (rank `{dominant['a']['partner_loading_rank']}/{dominant['b']['partner_loading_rank']}` among 53 candidates).","","Largest discovery-A sensitive-mode loadings:",""]
    for row in dominant["a"]["top_loadings"]: lines.append(f"- {row['candidate']}: `{row['loading']:+.3f}`")
    lines += ["","## Guardrails","","This geometry is destination- and corpus-conditioned. It is an ontology of substitutability for one target function, not an intrinsic semantic dictionary. The Fisher map is estimated by a small finite scale rather than an analytic JVP, although quadratic scaling was independently checked. Low rank would justify a compact compatibility basis; high rank would mean stable substitutability still requires many destination-specific coordinates."]
    REPORT.write_text("\n".join(lines)+"\n");print(json.dumps({"hypotheses":hypotheses,"summaries":summaries},indent=2));print(REPORT)


if __name__=="__main__":main()
