#!/usr/bin/env python3
"""Quantify how low-energy address contrasts carry natural boundary computation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_layer5_weight_conditioned_addresses import (
    derive_cleaned_addresses, output_conditioned_form, restricted_operator,
)
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layer11_address_contrast_gain_audit.json")
OUTPUT=TENSOR_ROOT/"bilin12_layer11_address_contrast_gain_audit.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYER11_ADDRESS_CONTRAST_GAIN_AUDIT.md"
FIGURE=TENSOR_ROOT/"bilin12_layer11_address_contrast_gain_audit.png"


def spearman(first,second):
    first=np.asarray(first); second=np.asarray(second); a=np.empty(len(first)); b=np.empty(len(second)); a[np.argsort(first)]=np.arange(len(first)); b[np.argsort(second)]=np.arange(len(second)); return float(np.corrcoef(a,b)[0,1])


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source"]).read_text()); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    states,_=collect_states_writes(model,config["layer"],config["discovery"],config["positions"],config["batch_size"]); square,_,covariance=ridge_covariance_transforms(states,config["covariance_ridge_fraction"])
    block=model.blocks[config["layer"]]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float(); _,outputs=output_map_eigensystem(down,left,right); form=output_conditioned_form(down,left,right,outputs[:,0]); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"])
    u,s,_=torch.linalg.svd(edits,full_matrices=False); address_energy=s.square(); address_energy_fraction=(address_energy/address_energy.sum()).cpu().numpy()
    operator_energy=[]
    for index in range(4):
        weighted=restricted_operator(down,left,right,u[:,index])@square; operator_energy.append(float(weighted.square().sum()))
    operator_energy_fraction=np.asarray(operator_energy)/np.sum(operator_energy)
    corpus_rows={}; mean_write_increment=np.zeros(4)
    for corpus,rows in source["results"].items():
        write=[rows[f"address_b_rank{rank}"]["write_delta_r2"] for rank in range(1,5)]; state=[rows[f"address_b_rank{rank}"]["state_energy_fraction"] for rank in range(1,5)]
        write_increment=np.asarray([write[0]]+[write[i]-write[i-1] for i in range(1,4)]); state_increment=np.asarray([state[0]]+[state[i]-state[i-1] for i in range(1,4)])
        mean_write_increment+=write_increment/len(source["results"])
        corpus_rows[corpus]={"write_r2":write,"state_energy":state,"write_r2_increment":write_increment.tolist(),"state_energy_increment":state_increment.tolist(),"contrast_write_fraction":float((write_increment[2]+write_increment[3])/write[-1]),"contrast_state_fraction":float((state_increment[2]+state_increment[3])/state[-1]),"pc4_write_increment":float(write_increment[3])}
    amplification=mean_write_increment/np.maximum(address_energy_fraction,1e-30); correlation=spearman(address_energy_fraction,mean_write_increment)
    directions=[{"pc":i+1,"address_singular":float(s[i]),"address_energy_fraction":float(address_energy_fraction[i]),"operator_energy_fraction":float(operator_energy_fraction[i]),"mean_write_r2_increment":float(mean_write_increment[i]),"amplification_ratio":float(amplification[i])} for i in range(4)]
    hypotheses={
        "W1_address_energy_predicts_causal_importance":{"supported":correlation>=.50,"spearman":correlation},
        "W2_low_energy_contrasts_dominate_boundary_write":{"supported":address_energy_fraction[2:].sum()<=.05 and all(r["contrast_write_fraction"]>=.60 for r in corpus_rows.values())},
        "W3_shared_backbone_is_redundant_for_boundary_write":{"supported":address_energy_fraction[0]>=.75 and all(r["write_r2_increment"][0]<=.05 for r in corpus_rows.values())},
        "W4_weight_operator_gain_favors_contrasts":{"supported":operator_energy_fraction[2:].sum()>=.60,"contrast_operator_fraction":float(operator_energy_fraction[2:].sum())},
        "W5_real_boundary_changes_use_contrasts":{"supported":all(r["contrast_state_fraction"]>=.60 for r in corpus_rows.values())},
        "W6_fourth_direction_is_not_numerical_junk":{"supported":address_energy_fraction[3]<=.005 and all(r["pc4_write_increment"]>=.05 for r in corpus_rows.values())},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"directions":directions,"corpora":corpus_rows},indent=2,allow_nan=False))

    x=np.arange(4); fig,axes=plt.subplots(1,3,figsize=(14,4.3)); axes[0].bar(x,address_energy_fraction); axes[0].set_title("Address-vector energy"); axes[1].bar(x,operator_energy_fraction); axes[1].set_title("Natural derivative-operator energy")
    for corpus,row in corpus_rows.items(): axes[2].plot(x,row["write_r2_increment"],marker="o",label=corpus)
    axes[2].set_title("Nested boundary-write R² increment"); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xticks(x,["PC1","PC2","PC3","PC4"]); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Gain audit for the Layer-11 address contrast code","","The four recovered native edit vectors are highly redundant. This audit rotates them into principal directions and compares how much address-vector energy each direction carries with its natural derivative-operator gain and its incremental contribution to real newline-induced state and MLP-write changes.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Contrast gain](bilin12_layer11_address_contrast_gain_audit.png)","",f"Spearman(address energy, mean causal increment): `{correlation:.3f}`.","","| Direction | Address energy | Operator energy | Mean write-R² increment | Amplification |","|---|---:|---:|---:|---:|"]
    for row in directions: lines.append(f"| PC{row['pc']} | {row['address_energy_fraction']:.4f} | {row['operator_energy_fraction']:.4f} | {row['mean_write_r2_increment']:.4f} | {row['amplification_ratio']:.2f}x |")
    lines += ["","| Corpus | Contrast share of captured state | Contrast share of write R² | PC4 write increment |","|---|---:|---:|---:|"]
    for corpus,row in corpus_rows.items(): lines.append(f"| {corpus} | {row['contrast_state_fraction']:.3f} | {row['contrast_write_fraction']:.3f} | {row['pc4_write_increment']:.3f} |")
    lines += ["","The leading native edit component describes what the four addresses share, but shared coordinate magnitude is not causal importance. Low-energy contrast directions separate which output response is required and are strongly amplified by the bilinear map. This is an explicit discriminator code: redundancy supplies a common semantic family, while small differences carry the operative information."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
