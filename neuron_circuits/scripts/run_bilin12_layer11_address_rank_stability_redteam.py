#!/usr/bin/env python3
"""Test rank and cross-covariance stability of natural Layer-11 boundary paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_layer11_boundary_natural_path_redteam import collect_boundary_deltas, evaluate_space
from run_bilin12_layer5_weight_conditioned_addresses import derive_cleaned_addresses, output_conditioned_form
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layer11_address_rank_stability_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin12_layer11_address_rank_stability_redteam.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYER11_ADDRESS_RANK_STABILITY_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin12_layer11_address_rank_stability_redteam.png"


@torch.inference_mode()
def addresses(model,spec,config):
    states,_=collect_states_writes(model,config["layer"],spec,config["covariance_positions"],config["batch_size"])
    square,_,metadata=ridge_covariance_transforms(states,config["covariance_ridge_fraction"])
    block=model.blocks[config["layer"]]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float(); _,outputs=output_map_eigensystem(down,left,right)
    form=output_conditioned_form(down,left,right,outputs[:,0]); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"])
    u,s,_=torch.linalg.svd(edits,full_matrices=False)
    return u,s,metadata,states


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    ua,sa,meta_a,states_a=addresses(model,config["covariance_a"],config); ub,sb,meta_b,states_b=addresses(model,config["covariance_b"],config)
    combined=torch.cat([states_a,states_b]); covariance=combined.T@combined/len(combined); _,state_vectors=torch.linalg.eigh(covariance)
    boundary_deltas=collect_boundary_deltas(model,tokenizer,config["boundary_delta_discovery"],config); _,_,delta_vh=torch.linalg.svd(boundary_deltas,full_matrices=False)
    spaces={}
    for rank in config["ranks"]:
        spaces[f"address_a_rank{rank}"]=ua[:,:rank]; spaces[f"address_b_rank{rank}"]=ub[:,:rank]; spaces[f"state_pca_rank{rank}"]=state_vectors[:,-rank:]; spaces[f"boundary_pca_rank{rank}"]=delta_vh[:rank].T
    results={corpus:evaluate_space(model,tokenizer,spec,config,spaces) for corpus,spec in config["evaluation"].items()}
    cross_slice={corpus:{"write_r2_difference":abs(results[corpus]["address_a_rank4"]["write_delta_r2"]-results[corpus]["address_b_rank4"]["write_delta_r2"]),"relative_kl_difference":abs(results[corpus]["address_a_rank4"]["relative_prediction_kl"]-results[corpus]["address_b_rank4"]["relative_prediction_kl"])} for corpus in results}
    hypotheses={
        "S1_rank3_retains_full_address_write":{"supported":all(results[c][f"address_{s}_rank3"]["write_delta_r2"]>=.9*results[c][f"address_{s}_rank4"]["write_delta_r2"] for c in results for s in ("a","b"))},
        "S2_full_address_function_is_cross_slice_stable":{"supported":all(r["write_r2_difference"]<=.05 and r["relative_kl_difference"]<=.10 for r in cross_slice.values()),"cross_slice":cross_slice},
        "S3_rank2_address_is_write_sufficient":{"supported":all(results[c][f"address_{s}_rank2"]["write_delta_r2"]>=.50 for c in results for s in ("a","b"))},
        "S4_rank1_address_is_not_write_sufficient":{"supported":all(results[c][f"address_{s}_rank1"]["write_delta_r2"]<.10 for c in results for s in ("a","b"))},
        "S5_weight_addresses_beat_state_pca_at_rank4":{"supported":all(results[c][f"address_{s}_rank4"]["write_delta_r2"]>results[c]["state_pca_rank4"]["write_delta_r2"] for c in results for s in ("a","b"))},
        "S6_weight_addresses_approach_supervised_boundary_pca":{"supported":all(results[c]["boundary_pca_rank4"]["write_delta_r2"]-results[c][f"address_{s}_rank4"]["write_delta_r2"]<=.10 for c in results for s in ("a","b"))},
        "S7_logit_sufficiency_remains_unresolved":{"supported":all(results[c][f"address_{s}_rank4"]["relative_prediction_kl"]>.25 for c in results for s in ("a","b"))},
        "S8_weak_address_directions_are_causally_necessary":{"supported":all(results[c][f"address_{s}_rank4"]["write_delta_r2"]-results[c][f"address_{s}_rank2"]["write_delta_r2"]>=.10 for c in results for s in ("a","b"))},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"address_singular_values":{"a":sa.cpu().tolist(),"b":sb.cpu().tolist()},"covariance":{"a":meta_a,"b":meta_b},"cross_slice":cross_slice,"results":results},indent=2,allow_nan=False))

    corpora=list(results); x=np.arange(len(config["ranks"])); fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for corpus in corpora:
        axes[0].plot(x,[results[corpus][f"address_b_rank{r}"]["write_delta_r2"] for r in config["ranks"]],marker="o",label=corpus)
        axes[1].plot(x,[results[corpus][f"address_b_rank{r}"]["relative_prediction_kl"] for r in config["ranks"]],marker="o",label=corpus)
    axes[0].set_title("Address-B write R² frontier"); axes[1].set_title("Address-B relative KL frontier")
    width=.12
    for j,name in enumerate(("address_a_rank4","address_b_rank4","state_pca_rank4","boundary_pca_rank4")):
        axes[2].bar(np.arange(len(corpora))+(j-1.5)*width,[results[c][name]["write_delta_r2"] for c in corpora],width,label=name)
    axes[2].set_xticks(np.arange(len(corpora)),corpora,rotation=20); axes[2].set_title("Rank-four controls"); axes[2].legend(fontsize=7)
    for ax in axes[:2]: ax.set_xticks(x,config["ranks"]); ax.set_xlabel("rank"); ax.legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Rank and cross-covariance stability of the Layer-11 natural boundary path","","The full address span captures real newline-induced MLP changes, but its native edit vectors are ill-conditioned and its weakest canonical direction is less stable. This follow-up compares rank-one through rank-four address PCA spaces from two independent covariance samples against matched natural-state and supervised boundary-delta PCA controls.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Rank stability](bilin12_layer11_address_rank_stability_redteam.png)","",f"Native address singular values, slice A/B: `{[round(x,4) for x in sa.cpu().tolist()]}` / `{[round(x,4) for x in sb.cpu().tolist()]}`.","","| Corpus/space | State energy | Write R² | Relative KL |","|---|---:|---:|---:|"]
    for corpus in corpora:
        for name in ("address_a_rank2","address_a_rank3","address_a_rank4","address_b_rank2","address_b_rank3","address_b_rank4","state_pca_rank4","boundary_pca_rank4"):
            row=results[corpus][name]; lines.append(f"| {corpus}/{name} | {row['state_energy_fraction']:.3f} | {row['write_delta_r2']:.3f} | {row['relative_prediction_kl']:.3f} |")
    lines += ["","Functional agreement across independently estimated spaces is stronger evidence than geometric canonical overlap alone. The rank frontier identifies whether the low-variance native edit directions are harmless redundancy or causally amplified coordinates. Behaviorally important weak directions are compatible with an ill-conditioned but reproducible tensor program."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
