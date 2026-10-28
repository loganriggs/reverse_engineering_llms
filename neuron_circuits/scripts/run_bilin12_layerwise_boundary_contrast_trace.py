#!/usr/bin/env python3
"""Trace and patch the emergence of Layer-11 boundary contrast coordinates."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_late_address_program_audit import exact_kl, load_rows
from run_bilin12_layer5_weight_conditioned_addresses import derive_cleaned_addresses, output_conditioned_form
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layerwise_boundary_contrast_trace.json")
OUTPUT=TENSOR_ROOT/"bilin12_layerwise_boundary_contrast_trace.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYERWISE_BOUNDARY_CONTRAST_TRACE.md"
FIGURE=TENSOR_ROOT/"bilin12_layerwise_boundary_contrast_trace.png"


@torch.inference_mode()
def trace(model,ids):
    dimension=model.config.n_embd; x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; bus=None; rows=[]
    for block in model.blocks:
        mixed=block.lambdas[0]*x+block.lambdas[1]*x0; attention,bus=block.attn(F.rms_norm(mixed,(dimension,)),bus); pre=mixed+attention; state=F.rms_norm(pre,(dimension,)); write=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias; xout=pre+write
        rows.append({"mixed":mixed,"attention":attention,"state":state,"write":write,"x":xout,"bus":bus}); x=xout
    raw=model.lm_head(F.rms_norm(x,(dimension,))[:,-1]).float(); logits=30*torch.tanh(raw/30)
    return x0,rows,logits


@torch.inference_mode()
def run_patched_suffix(model,x,bus,x0,start_layer):
    dimension=model.config.n_embd
    for block in model.blocks[start_layer:11]: x,bus=block(x,bus,x0)
    block=model.blocks[11]; mixed=block.lambdas[0]*x+block.lambdas[1]*x0; attention,bus=block.attn(F.rms_norm(mixed,(dimension,)),bus); pre=mixed+attention; state=F.rms_norm(pre,(dimension,)); write=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias; final=pre+write; raw=model.lm_head(F.rms_norm(final,(dimension,))[:,-1]).float(); logits=30*torch.tanh(raw/30)
    return state[:,-1],logits


@torch.inference_mode()
def evaluate(model,tokenizer,spec,config,contrast_basis):
    rows=load_rows(spec); newline=tokenizer.encode("\n",add_special_tokens=False)[0]; layers=len(model.blocks); store=defaultdict(float); examples=0
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); original=torch.as_tensor(raw,device="cuda"); keep=original[:,config["position"]]!=newline
        if not bool(keep.any()): continue
        original=original[keep]; changed=original.clone(); changed[:,config["position"]]=newline; examples+=len(original)
        x0,o,base_logits=trace(model,original); x0c,c,changed_logits=trace(model,changed); position=config["position"]
        final_delta=(c[11]["state"][:,position]-o[11]["state"][:,position])@contrast_basis; final_den=float(final_delta.square().sum()); store["final_contrast_den"]+=final_den
        full_kl=exact_kl(base_logits,changed_logits); store["full_kl"]+=float(full_kl.sum())
        for layer in range(layers):
            state_delta=(c[layer]["state"][:,position]-o[layer]["state"][:,position])@contrast_basis; store[f"layer{layer}/state_contrast_energy"]+=float(state_delta.square().sum())
            output_delta=c[layer]["x"][:,position]-o[layer]["x"][:,position]; output_projection=output_delta@contrast_basis; denominator=float(output_projection.square().sum()); store[f"layer{layer}/output_den"]+=denominator
            components={"mixed":c[layer]["mixed"][:,position]-o[layer]["mixed"][:,position],"attention":c[layer]["attention"][:,position]-o[layer]["attention"][:,position],"mlp":c[layer]["write"][:,position]-o[layer]["write"][:,position]}
            for name,value in components.items(): store[f"layer{layer}/{name}_attribution"]+=float(((value@contrast_basis)*output_projection).sum())
            if layer<11:
                delta_x=c[layer]["x"][:,position]-o[layer]["x"][:,position]; projected=(delta_x@contrast_basis)@contrast_basis.T; patched=o[layer]["x"].clone(); patched[:,position]+=projected
                patched_state,patched_logits=run_patched_suffix(model,patched,o[layer]["bus"],x0,layer+1); patched_coord=(patched_state-o[11]["state"][:,position])@contrast_basis
                store[f"layer{layer}/patch_error"]+=float((final_delta-patched_coord).square().sum()); store[f"layer{layer}/patch_prediction_kl"]+=float(exact_kl(changed_logits,patched_logits).sum())
    result={"examples":examples,"layers":{}}
    final_den=store["final_contrast_den"]; full_kl=store["full_kl"]
    for layer in range(layers):
        denominator=store[f"layer{layer}/output_den"]
        row={"contrast_energy_ratio":store[f"layer{layer}/state_contrast_energy"]/max(final_den,1e-30),"mixed_attribution":store[f"layer{layer}/mixed_attribution"]/max(denominator,1e-30),"attention_attribution":store[f"layer{layer}/attention_attribution"]/max(denominator,1e-30),"mlp_attribution":store[f"layer{layer}/mlp_attribution"]/max(denominator,1e-30)}
        if layer<11:
            row["contrast_patch_recovery"]=1-store[f"layer{layer}/patch_error"]/max(final_den,1e-30); row["relative_full_token_kl"]=store[f"layer{layer}/patch_prediction_kl"]/max(full_kl,1e-30)
        result["layers"][str(layer)]=row
    return result


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    states,_=collect_states_writes(model,config["address_layer"],config["covariance_discovery"],config["covariance_positions"],config["batch_size"]); square,_,covariance=ridge_covariance_transforms(states,config["covariance_ridge_fraction"]); block=model.blocks[config["address_layer"]]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float(); _,outputs=output_map_eigensystem(down,left,right); form=output_conditioned_form(down,left,right,outputs[:,0]); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"]); u,_,_=torch.linalg.svd(edits,full_matrices=False); contrast_basis=u[:,2:4]
    results={corpus:evaluate(model,tokenizer,spec,config,contrast_basis) for corpus,spec in config["evaluation"].items()}
    first_half={corpus:next((layer for layer in range(12) if row["layers"][str(layer)]["contrast_energy_ratio"]>=.5),None) for corpus,row in results.items()}
    early_patch={corpus:max(row["layers"][str(layer)]["contrast_patch_recovery"] for layer in range(9)) for corpus,row in results.items()}
    best_layer={corpus:max(range(11),key=lambda layer:row["layers"][str(layer)]["contrast_patch_recovery"]) for corpus,row in results.items()}
    largest_mlp={corpus:max(range(12),key=lambda layer:row["layers"][str(layer)]["mlp_attribution"]) for corpus,row in results.items()}
    hypotheses={
        "E1_boundary_contrast_emerges_before_layer9":{"supported":all(any(row["layers"][str(layer)]["contrast_energy_ratio"]>=.5 for layer in range(9)) for row in results.values()),"first_half_layers":first_half},
        "E2_early_contrast_patch_is_causally_effective":{"supported":all(value>=.5 for value in early_patch.values()),"best_early_recovery":early_patch},
        "E3_layer10_contrast_patch_is_nearly_sufficient":{"supported":all(row["layers"]["10"]["contrast_patch_recovery"]>=.8 for row in results.values())},
        "E4_contrast_only_patch_is_not_full_logit_sufficient":{"supported":all(min(row["layers"][str(layer)]["relative_full_token_kl"] for layer in range(11))>.25 for row in results.values()),"best_layers":best_layer},
        "E5_middle_mlp_writes_create_the_largest_contrast_increment":{"supported":all(4<=layer<=8 for layer in largest_mlp.values()),"largest_mlp_layers":largest_mlp},
        "E6_attention_never_dominates_direct_contrast_writing":{"supported":all(row["layers"][str(layer)]["attention_attribution"]<=.5 for row in results.values() for layer in range(12))},
        "E7_emergence_depth_transfers_ood":{"supported":None not in first_half.values() and max(first_half.values())-min(first_half.values())<=2},
        "E8_late_residual_patch_recovery_is_monotone":{"supported":all(all(row["layers"][str(layer+1)]["contrast_patch_recovery"]>=row["layers"][str(layer)]["contrast_patch_recovery"] for layer in range(7,10)) for row in results.values())},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"results":results},indent=2,allow_nan=False))

    layers=np.arange(12); fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for corpus,row in results.items(): axes[0].plot(layers,[row["layers"][str(l)]["contrast_energy_ratio"] for l in layers],marker="o",label=corpus); axes[1].plot(layers[:11],[row["layers"][str(l)]["contrast_patch_recovery"] for l in layers[:11]],marker="o",label=corpus)
    axes[0].axhline(.5,color="black",ls="--",lw=1); axes[0].set_title("Contrast emergence"); axes[1].axhline(.5,color="black",ls="--",lw=1); axes[1].set_title("Contrast-only patch recovery")
    mean_mlp=[np.mean([row["layers"][str(l)]["mlp_attribution"] for row in results.values()]) for l in layers]; mean_attn=[np.mean([row["layers"][str(l)]["attention_attribution"] for row in results.values()]) for l in layers]; axes[2].bar(layers-.18,mean_mlp,.36,label="MLP"); axes[2].bar(layers+.18,mean_attn,.36,label="attention"); axes[2].set_title("Direct contrast attribution")
    for ax in axes: ax.set_xlabel("layer"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Layerwise emergence and causal tracing of boundary contrasts","","For actual final-token-to-newline edits, this audit tracks the final Layer-11 address PC3–PC4 coordinates through every normalized pre-MLP state. At each block, the exact output-residual contrast is allocated among mixed residual/embedding input, attention, and MLP write. A stronger intervention patches only the two contrast coordinates of each changed layer output into the original run, keeps the original attention value bus, and measures recovery of the final Layer-11 contrast and full changed logits.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Layerwise trace](bilin12_layerwise_boundary_contrast_trace.png)","","| Corpus | First ≥50% energy | Best early patch | Layer-10 recovery | Best patch layer | Best relative full KL | Largest MLP layer |","|---|---:|---:|---:|---:|---:|---:|"]
    for corpus,row in results.items():
        best=best_layer[corpus]; lines.append(f"| {corpus} | {first_half[corpus]} | {early_patch[corpus]:.3f} | {row['layers']['10']['contrast_patch_recovery']:.3f} | {best} | {row['layers'][str(best)]['relative_full_token_kl']:.3f} | {largest_mlp[corpus]} |")
    lines += ["","Descriptive emergence and causal patch recovery answer different questions. A direction can be visible early yet not survive downstream processing when patched alone. Keeping the original value bus makes this specifically a residual-route test; it does not claim that attention's separate value-memory route is absent."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
