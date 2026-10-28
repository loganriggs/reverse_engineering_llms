#!/usr/bin/env python3
"""Test whether real newline-induced state changes use recovered Layer-11 addresses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_late_address_program_audit import exact_kl, load_rows, run_suffix
from run_bilin12_layer5_weight_conditioned_addresses import derive_cleaned_addresses, output_conditioned_form
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin12_layer11_boundary_natural_path_redteam.json")
OUTPUT=TENSOR_ROOT/"bilin12_layer11_boundary_natural_path_redteam.json"
REPORT=TENSOR_ROOT/"BILIN12_LAYER11_BOUNDARY_NATURAL_PATH_REDTEAM.md"
FIGURE=TENSOR_ROOT/"bilin12_layer11_boundary_natural_path_redteam.png"


@torch.inference_mode()
def layer_context(model,ids,layer):
    dimension=model.config.n_embd; x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; value_bus=None
    for block in model.blocks[:layer]: x,value_bus=block(x,value_bus,x0)
    block=model.blocks[layer]; mixed=block.lambdas[0]*x+block.lambdas[1]*x0
    attention,value_bus=block.attn(F.rms_norm(mixed,(dimension,)),value_bus); pre=mixed+attention
    state=F.rms_norm(pre,(dimension,))
    return x0,pre,state,value_bus


@torch.inference_mode()
def collect_boundary_deltas(model,tokenizer,spec,config):
    rows=load_rows(spec); newline=tokenizer.encode("\n",add_special_tokens=False)
    if len(newline)!=1: raise RuntimeError(f"newline not one token: {newline}")
    deltas=[]
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); original=torch.as_tensor(raw,device="cuda")
        keep=original[:,config["position"]]!=newline[0]
        if not bool(keep.any()): continue
        original=original[keep]; changed=original.clone(); changed[:,config["position"]]=newline[0]
        _,_,state,_=layer_context(model,original,config["layer"]); _,_,changed_state,_=layer_context(model,changed,config["layer"])
        deltas.append(changed_state[:,config["position"]]-state[:,config["position"]])
    return torch.cat(deltas)


def projection(delta,basis):
    return (delta@basis)@basis.T


@torch.inference_mode()
def evaluate_space(model,tokenizer,spec,config,spaces):
    rows=load_rows(spec); newline=tokenizer.encode("\n",add_special_tokens=False)[0]
    block=model.blocks[config["layer"]]; store={name:{"state_num":0.,"state_den":0.,"write_num":0.,"write_den":0.,"prediction_kl":[],"mlp_kl":[],"full_kl":[]} for name in spaces}
    for start in range(0,len(rows),config["batch_size"]):
        raw=np.asarray(rows[start:start+config["batch_size"],:128],dtype=np.int64); original=torch.as_tensor(raw,device="cuda")
        keep=original[:,config["position"]]!=newline
        if not bool(keep.any()): continue
        original=original[keep]; changed=original.clone(); changed[:,config["position"]]=newline
        x0,pre,state,value_bus=layer_context(model,original,config["layer"]); _,changed_pre,changed_state,changed_bus=layer_context(model,changed,config["layer"])
        old=state[:,config["position"]]; new=changed_state[:,config["position"]]; delta=new-old
        old_write=block.mlp.Down(block.mlp.Left(old)*block.mlp.Right(old))+block.mlp.Down_bias
        new_write=block.mlp.Down(block.mlp.Left(new)*block.mlp.Right(new))+block.mlp.Down_bias; write_delta=new_write-old_write
        native_write=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
        native_x=pre+native_write; baseline=run_suffix(model,native_x,value_bus,x0,config["layer"]+1)
        mlp_x=native_x.clone(); mlp_x[:,config["position"]]=pre[:,config["position"]]+new_write
        exact_mlp=run_suffix(model,mlp_x,value_bus,x0,config["layer"]+1)
        changed_write=block.mlp.Down(block.mlp.Left(changed_state)*block.mlp.Right(changed_state))+block.mlp.Down_bias
        full_changed=run_suffix(model,changed_pre+changed_write,changed_bus,F.rms_norm(model.embedding(changed),(model.config.n_embd,)),config["layer"]+1)
        mlp_kl=exact_kl(baseline,exact_mlp); full_kl=exact_kl(baseline,full_changed)
        for name,basis in spaces.items():
            projected=projection(delta,basis); predicted_write=block.mlp.Down(block.mlp.Left(old+projected)*block.mlp.Right(old+projected))+block.mlp.Down_bias
            predicted_delta=predicted_write-old_write
            predicted_x=native_x.clone(); predicted_x[:,config["position"]]=pre[:,config["position"]]+predicted_write
            predicted_logits=run_suffix(model,predicted_x,value_bus,x0,config["layer"]+1)
            row=store[name]; row["state_num"]+=float((delta-projected).square().sum()); row["state_den"]+=float(delta.square().sum()); row["write_num"]+=float((write_delta-predicted_delta).square().sum()); row["write_den"]+=float(write_delta.square().sum()); row["prediction_kl"].extend(exact_kl(exact_mlp,predicted_logits).cpu().tolist()); row["mlp_kl"].extend(mlp_kl.cpu().tolist()); row["full_kl"].extend(full_kl.cpu().tolist())
    result={}
    for name,row in store.items():
        mean_mlp=float(np.mean(row["mlp_kl"])); mean_full=float(np.mean(row["full_kl"])); mean_prediction=float(np.mean(row["prediction_kl"]))
        result[name]={"state_energy_fraction":1-row["state_num"]/max(row["state_den"],1e-30),"write_delta_r2":1-row["write_num"]/max(row["write_den"],1e-30),"mean_mlp_path_kl":mean_mlp,"mean_full_token_kl":mean_full,"mlp_to_full_kl_ratio":mean_mlp/max(mean_full,1e-30),"mean_prediction_kl":mean_prediction,"relative_prediction_kl":mean_prediction/max(mean_mlp,1e-30)}
    return result


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); model=load_snapshot_model(Path(config["snapshot"]),"cuda")
    states,_=collect_states_writes(model,config["layer"],config["covariance_discovery"],config["covariance_positions"],config["batch_size"])
    square,_,covariance=ridge_covariance_transforms(states,config["covariance_ridge_fraction"]); block=model.blocks[config["layer"]]; down=block.mlp.Down.weight.float(); left=block.mlp.Left.weight.float(); right=block.mlp.Right.weight.float(); _,outputs=output_map_eigensystem(down,left,right); form=output_conditioned_form(down,left,right,outputs[:,0]); _,_,edits=derive_cleaned_addresses(form,square,config["address_count"])
    address_span=torch.linalg.qr(edits,mode="reduced").Q; edit_values,edit_vectors=torch.linalg.eigh(edits@edits.T); address_backbone=edit_vectors[:,-1:]
    state_cov=states.T@states/len(states); _,state_vectors=torch.linalg.eigh(state_cov); state_pca4=state_vectors[:,-4:]
    boundary_deltas=collect_boundary_deltas(model,tokenizer,config["boundary_delta_discovery"],config); _,_,vh=torch.linalg.svd(boundary_deltas,full_matrices=False); boundary_pca4=vh[:4].T
    generator=torch.Generator(device="cuda").manual_seed(config["seed"]); spaces={"address_span":address_span,"address_backbone":address_backbone,"state_pca4":state_pca4,"boundary_pca4":boundary_pca4}
    for index in range(config["random_replicates"]): spaces[f"random4_{index}"]=torch.linalg.qr(torch.randn(len(square),4,generator=generator,device="cuda"),mode="reduced").Q
    results={corpus:evaluate_space(model,tokenizer,spec,config,spaces) for corpus,spec in config["evaluation"].items()}
    for corpus in results:
        random_rows=[results[corpus][f"random4_{i}"] for i in range(config["random_replicates"])]
        results[corpus]["random4"]={key:float(np.median([row[key] for row in random_rows])) for key in random_rows[0]}
        for i in range(config["random_replicates"]): del results[corpus][f"random4_{i}"]
    hypotheses={
        "N1_boundary_changes_enter_address_span":{"supported":all(results[c]["address_span"]["state_energy_fraction"]>=2*results[c]["random4"]["state_energy_fraction"] for c in results)},
        "N2_address_span_captures_boundary_mlp_write":{"supported":all(results[c]["address_span"]["write_delta_r2"]>=.50 for c in results)},
        "N3_address_span_predicts_boundary_mlp_logits":{"supported":all(results[c]["address_span"]["relative_prediction_kl"]<=.25 for c in results)},
        "N4_shared_address_backbone_is_sufficient":{"supported":all(results[c]["address_backbone"]["write_delta_r2"]>=.75*results[c]["address_span"]["write_delta_r2"] for c in results)},
        "N5_address_span_beats_unsupervised_state_pca":{"supported":all(results[c]["address_span"]["write_delta_r2"]>results[c]["state_pca4"]["write_delta_r2"] for c in results)},
        "N6_supervised_boundary_pca_is_an_upper_bound":{"supported":all(results[c]["boundary_pca4"]["write_delta_r2"]>results[c]["address_span"]["write_delta_r2"] for c in results)},
        "N7_layer11_mlp_path_is_material_to_full_token_edit":{"supported":all(results[c]["address_span"]["mlp_to_full_kl_ratio"]>=.10 for c in results)},
        "N8_address_advantage_transfers_ood":{"supported":all(results[c]["address_span"]["write_delta_r2"]>0 and results[c]["address_span"]["write_delta_r2"]>results[c]["random4"]["write_delta_r2"] for c in ("wikitext","tinystories"))},
    }
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"covariance":covariance,"address_edit_gram_eigenvalues":torch.linalg.eigvalsh(edits.T@edits).cpu().tolist(),"boundary_delta_singular_values":torch.linalg.svdvals(boundary_deltas).cpu().tolist()[:16],"results":results},indent=2,allow_nan=False))

    corpora=list(results); spaces_to_plot=["address_span","address_backbone","state_pca4","boundary_pca4","random4"]; x=np.arange(len(corpora)); fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for name in spaces_to_plot:
        axes[0].plot(x,[results[c][name]["state_energy_fraction"] for c in corpora],marker="o",label=name); axes[1].plot(x,[results[c][name]["write_delta_r2"] for c in corpora],marker="o",label=name); axes[2].plot(x,[results[c][name]["relative_prediction_kl"] for c in corpora],marker="o",label=name)
    axes[0].set_title("Boundary state-change coverage"); axes[1].set_title("MLP write-delta R²"); axes[2].set_title("Relative logit prediction KL")
    for ax in axes: ax.set_xticks(x,corpora,rotation=20); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE,dpi=180); plt.close(fig)

    lines=["# Natural-path red team for the Layer-11 boundary program","","The recovered addresses define compact derivatives whose context covectors detect boundaries. This test asks the stronger question: do real ordinary-token-to-newline state changes enter the address span, and does that span reproduce the resulting MLP write and final-logit effects? Random four-spaces, natural-state PCA, and a separately fitted boundary-delta PCA are controls.","","## Locked hypotheses",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Natural path](bilin12_layer11_boundary_natural_path_redteam.png)","","| Corpus/space | State energy | Write R² | Relative prediction KL | MLP/full KL |","|---|---:|---:|---:|---:|"]
    for corpus in corpora:
        for name in spaces_to_plot:
            row=results[corpus][name]; lines.append(f"| {corpus}/{name} | {row['state_energy_fraction']:.3f} | {row['write_delta_r2']:.3f} | {row['relative_prediction_kl']:.3f} | {row['mlp_to_full_kl_ratio']:.3f} |")
    lines += ["","A derivative can have a clean semantic context variable without lying on the natural causal path that produces the corresponding token transition. State coverage asks whether the upstream change enters the recovered tangent directions; exact projected MLP recomputation then tests nonlinear write and downstream behavior. The supervised boundary PCA is an upper bound for this specific edit, not a weight-only explanation."]
    REPORT.write_text("\n".join(lines)+"\n"); print(REPORT)


if __name__=="__main__": main()
