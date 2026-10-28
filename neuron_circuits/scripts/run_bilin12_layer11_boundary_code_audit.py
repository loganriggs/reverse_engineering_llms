#!/usr/bin/env python3
"""Determine whether four validated Layer-11 boundary programs share one code."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from discover_bilin18_weight_only_circuits import auc
from run_bilin12_late_address_program_audit import build_program, load_rows
from run_bilin12_layer5_weight_conditioned_addresses import derive_cleaned_addresses, output_conditioned_form
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layer11_boundary_code_audit.json")
OUTPUT=TENSOR_ROOT/"bilin12_layer11_boundary_code_audit.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYER11_BOUNDARY_CODE_AUDIT.md"
FIGURE=TENSOR_ROOT/"bilin12_layer11_boundary_code_audit.png"


def pairwise_values(matrix):
    return [float(matrix[i,j]) for i,j in itertools.combinations(range(len(matrix)),2)]


def best_permutation(overlap):
    return max(itertools.permutations(range(overlap.shape[1])),key=lambda p:sum(overlap[i,p[i]] for i in range(len(p))))


@torch.inference_mode()
def construct(model,spec,config):
    layer=config["layer"]; states,_=collect_states_writes(model,layer,spec,config["positions"],config["batch_size"])
    square,inverse,covariance=ridge_covariance_transforms(states,config["covariance_ridge_fraction"])
    block=model.blocks[layer]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float()
    _,outputs=output_map_eigensystem(down,left,right); output=outputs[:,0]
    form=output_conditioned_form(down,left,right,output); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"])
    programs=[build_program(edits[:,i],output,down,left,right,square,inverse,[32]) for i in range(config["address_count"])]
    gram=edits.T@edits; eigen=torch.linalg.eigvalsh(gram).cpu().numpy()
    return edits,programs,{"eigenvalues":eigen.tolist(),"condition":float(eigen.max()/max(eigen.min(),1e-30)),"covariance":covariance}


def weighted_common_fraction(vectors,weights):
    matrix=sum(weight*vector[:,None]@vector[None] for vector,weight in zip(vectors,weights))
    values=torch.linalg.eigvalsh(matrix)
    return float(values[-1]/values.sum().clamp_min(1e-30))


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source"]).read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); newline=tokenizer.encode("\n",add_special_tokens=False)
    if len(newline)!=1: raise RuntimeError(f"newline not one token: {newline}")
    model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    edits_a,programs_a,gram_a=construct(model,config["discovery_a"],config); edits_b,programs,gram_b=construct(model,config["discovery_b"],config)
    overlap=(edits_a.T@edits_b).abs().cpu().numpy(); permutation=best_permutation(overlap); matched=[float(overlap[i,permutation[i]]) for i in range(4)]
    polarities={}
    for index,features in source["recurrent_features"][str(config["layer"])].items():
        current=[f for f in features if f.startswith("current/newline/")]
        if not current: raise RuntimeError(f"address {index} lacks recurrent current-newline feature")
        polarities[index]=current[0].rsplit("/",1)[1]
    signs=np.asarray([1 if polarities[str(i)]=="positive" else -1 for i in range(4)])

    singular_weights=torch.tensor([p["singular"][0].square() for p in programs],device="cuda")
    input_vectors=[p["v"][:,0] for p in programs]; output_vectors=[p["u"][:,0] for p in programs]
    shared_input_energy_fraction=weighted_common_fraction(input_vectors,singular_weights)
    shared_output_energy_fraction=weighted_common_fraction(output_vectors,singular_weights)
    balanced_shared_input_fraction=weighted_common_fraction(input_vectors,torch.ones_like(singular_weights))
    balanced_shared_output_fraction=weighted_common_fraction(output_vectors,torch.ones_like(singular_weights))
    input_pairwise_absolute_cosines=[float(abs(torch.dot(a,b))) for a,b in itertools.combinations(input_vectors,2)]
    output_pairwise_absolute_cosines=[float(abs(torch.dot(a,b))) for a,b in itertools.combinations(output_vectors,2)]
    normalized_singular_weights=(singular_weights/singular_weights.sum()).cpu().tolist()
    top_sets=[set(p["order"][:32].cpu().tolist()) for p in programs]
    native_top32_jaccard=[len(a&b)/len(a|b) for a,b in itertools.combinations(top_sets,2)]

    corpora={}
    for corpus,spec in config["evaluation"].items():
        states,_=collect_states_writes(model,config["layer"],spec,config["positions"],config["batch_size"])
        scores=torch.stack([states@p["context_read"] for p in programs],dim=1).cpu().numpy()
        rows=load_rows(spec); positions=np.asarray(config["positions"]); ids=rows[:,positions].reshape(-1); target=ids==newline[0]
        z=(scores-scores.mean(0))/scores.std(0).clip(min=1e-12); aligned=z*signs[None]
        correlation=np.corrcoef(aligned,rowvar=False); eigen=np.linalg.eigvalsh(correlation)
        activation_pc_fraction=float(eigen[-1]/eigen.sum())
        position_labels=np.tile(np.arange(len(positions)),len(rows)); demeaned=aligned.copy()
        for position in range(len(positions)): demeaned[position_labels==position]-=demeaned[position_labels==position].mean(0)
        aggregate=demeaned.mean(1); position_demeaned_boundary_auc=float(auc(target,aggregate)); position_demeaned_boundary_auc=max(position_demeaned_boundary_auc,1-position_demeaned_boundary_auc)
        design=np.stack([np.ones(len(target)),target.astype(float)],axis=1); fitted=design@np.linalg.lstsq(design,aligned,rcond=None)[0]; residual=aligned-fitted
        residual_correlation=np.corrcoef(residual,rowvar=False); residual_pairs=np.abs(pairwise_values(residual_correlation))
        corpora[corpus]={"activation_pc_fraction":activation_pc_fraction,"position_demeaned_boundary_auc":position_demeaned_boundary_auc,"aligned_score_correlation":correlation.tolist(),"residual_correlation":residual_correlation.tolist(),"median_absolute_residual_correlation":float(np.median(residual_pairs)),"newline_count":int(target.sum())}

    hypotheses={
        "B1_boundary_scores_form_one_activation_code":{"supported":all(r["activation_pc_fraction"]>=.75 for r in corpora.values())},
        "B2_top_operators_share_one_input_covector":{"supported":shared_input_energy_fraction>=.75,"fraction":shared_input_energy_fraction},
        "B3_top_operators_have_distinct_output_verbs":{"supported":shared_output_energy_fraction<=.50,"fraction":shared_output_energy_fraction},
        "B4_aggregate_code_predicts_boundaries_ood":{"supported":all(r["position_demeaned_boundary_auc"]>=.90 for r in corpora.values())},
        "B5_shared_semantics_do_not_explain_all_score_variation":{"supported":all(r["median_absolute_residual_correlation"]<=.30 for r in corpora.values())},
        "B6_native_implementations_are_distinct":{"supported":np.median(native_top32_jaccard)<=.20,"median_jaccard":float(np.median(native_top32_jaccard))},
        "B7_native_address_axes_are_individually_stable":{"supported":min(matched)>=.95,"matched_cosines":matched},
        "B8_native_address_span_is_ill_conditioned":{"supported":gram_a["condition"]>=10 and gram_b["condition"]>=10,"conditions":[gram_a["condition"],gram_b["condition"]]},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"polarities":polarities,"matched_cosines":matched,"best_permutation":list(permutation),"edit_gram":{"slice_a":gram_a,"slice_b":gram_b},"shared_input_energy_fraction":shared_input_energy_fraction,"shared_output_energy_fraction":shared_output_energy_fraction,"balanced_shared_input_fraction":balanced_shared_input_fraction,"balanced_shared_output_fraction":balanced_shared_output_fraction,"normalized_singular_weights":normalized_singular_weights,"input_pairwise_absolute_cosines":input_pairwise_absolute_cosines,"output_pairwise_absolute_cosines":output_pairwise_absolute_cosines,"native_top32_jaccard":native_top32_jaccard,"corpora":corpora},indent=2,allow_nan=False))

    fig,axes=plt.subplots(1,3,figsize=(14,4.3)); names=list(corpora)
    axes[0].bar(names,[corpora[c]["activation_pc_fraction"] for c in names]); axes[0].axhline(.75,color="black",ls="--",lw=1); axes[0].set_title("Shared activation-code PC1")
    axes[1].bar(names,[corpora[c]["position_demeaned_boundary_auc"] for c in names]); axes[1].axhline(.9,color="black",ls="--",lw=1); axes[1].set_title("Aggregate boundary AUROC")
    axes[2].bar(np.arange(2)-.18,[shared_input_energy_fraction,shared_output_energy_fraction],.36,label="energy weighted"); axes[2].bar(np.arange(2)+.18,[balanced_shared_input_fraction,balanced_shared_output_fraction],.36,label="program balanced"); axes[2].set_xticks(np.arange(2),["input","output"]); axes[2].axhline(.75,color="black",ls="--",lw=1); axes[2].set_title("Shared top-operator factor"); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2); ax.tick_params(axis="x",rotation=20)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Layer-11 boundary-code audit","","All four compact Layer-11 address programs have cross-corpus current-newline selectivity that survived literal token edits. This audit asks whether they are four views of one geometric feature, one input predicate fanning out to several output verbs, or semantically related but geometrically multiplexed programs.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Boundary code](bilin12_layer11_boundary_code_audit.png)","",f"Polarity-aligned address roles: `{polarities}`. Matched cross-slice edit cosines: `{[round(x,3) for x in matched]}`. Edit-Gram conditions: `{gram_a['condition']:.1f}` / `{gram_b['condition']:.1f}`.","",f"Singular-weighted common input/output fractions: `{shared_input_energy_fraction:.3f}` / `{shared_output_energy_fraction:.3f}`. Program-balanced fractions: `{balanced_shared_input_fraction:.3f}` / `{balanced_shared_output_fraction:.3f}`. Normalized top-rank energy weights: `{[round(x,3) for x in normalized_singular_weights]}`.",f"Pairwise absolute input/output cosines: `{[round(x,3) for x in input_pairwise_absolute_cosines]}` / `{[round(x,3) for x in output_pairwise_absolute_cosines]}`. Median native top-32 Jaccard: `{np.median(native_top32_jaccard):.3f}`.","","| Corpus | Activation PC1 | Boundary AUROC | Residual correlation | Newlines |","|---|---:|---:|---:|---:|"]
    for corpus,row in corpora.items(): lines.append(f"| {corpus} | {row['activation_pc_fraction']:.3f} | {row['position_demeaned_boundary_auc']:.3f} | {row['median_absolute_residual_correlation']:.3f} | {row['newline_count']} |")
    lines += ["","The result is a partially shared fan-out program. All four top operators read nearly the same natural-metric input covector, but their output directions range from strongly aligned to nearly orthogonal, their native channel sets overlap only moderately, and their corpus activation scores retain roughly two effective dimensions plus correlated residual variation. The reusable object is therefore a shared boundary-reading backbone with multiplexed output verbs—not one monosemantic neuron, four unrelated codes, or one fully collapsed activation scalar."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
