#!/usr/bin/env python3
"""Factor exact Layer-8 tensors restricted to routed sensitivity addresses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from transformers import AutoTokenizer


CONFIG=Path("configs/bilin18_layer8_restricted_tensor_program.json")
SOURCE=TENSOR_ROOT/"layer8_multimetric_quantization.npz"
OUTPUT=TENSOR_ROOT/"layer8_restricted_tensor_program.json"
ARRAYS=TENSOR_ROOT/"layer8_restricted_tensor_program.npz"
REPORT=TENSOR_ROOT/"LAYER8_RESTRICTED_TENSOR_PROGRAM.md"
FIGURE=TENSOR_ROOT/"layer8_restricted_tensor_program.png"


def restricted_tensor(down,left,right,edit_directions):
    left_edit=left@edit_directions;right_edit=right@edit_directions
    return torch.einsum("oh,ha,hj->oaj",down,left_edit,right)+torch.einsum("oh,ha,hj->oaj",down,right_edit,left)


def tucker_prediction(whitened,weighted_tensor,output_basis,context_basis,rank):
    output=output_basis[:,:rank];context=context_basis[:,:rank];core=torch.einsum("os,oaj,jr->sar",output,weighted_tensor,context);coordinates=whitened@context
    return torch.einsum("nr,sar,os->nao",coordinates,core,output)


def energy_fraction(singular,rank):
    energy=singular.square();return float(energy[:rank].sum()/energy.sum().clamp_min(1e-30))


def prediction_r2(target,prediction):
    return float(1-(target-prediction).square().sum()/target.square().sum().clamp_min(1e-30))


def ridge_covariance_transforms(x, ridge_fraction):
    """Return C^(1/2) and C^(-1/2) for a trace-scaled ridge covariance.

    Routed panels can contain far fewer rows than the residual dimension.  A
    median-eigenvalue floor is then zero (up to roundoff), so it neither defines
    a metric off the observed span nor gives a stable inverse.  The explicit
    ridge is part of the explanatory metric and is reported rather than hidden
    as a numerical patch.
    """
    dimension = x.shape[1]
    empirical = x.T @ x / len(x)
    mean_variance = torch.trace(empirical) / dimension
    ridge = mean_variance * ridge_fraction
    covariance = 0.5 * (empirical + empirical.T)
    covariance = covariance + ridge * torch.eye(
        dimension, device=x.device, dtype=x.dtype
    )
    try:
        values, vectors = torch.linalg.eigh(covariance)
    except torch._C._LinAlgError:
        values, vectors = torch.linalg.eigh(covariance.double().cpu())
        values = values.to(device=x.device, dtype=x.dtype)
        vectors = vectors.to(device=x.device, dtype=x.dtype)
    values = values.clamp_min(ridge)
    square = (vectors * values.sqrt()[None]) @ vectors.T
    inverse = (vectors * values.rsqrt()[None]) @ vectors.T
    return square, inverse, {
        "estimator": "uncentered_empirical_plus_trace_scaled_isotropic_ridge",
        "rows": int(len(x)),
        "dimension": int(dimension),
        "ridge_fraction": float(ridge_fraction),
        "ridge_value": float(ridge),
        "mean_empirical_variance": float(mean_variance),
        "minimum_eigenvalue": float(values.min()),
        "maximum_eigenvalue": float(values.max()),
        "condition_number": float(values.max() / values.min()),
    }


def gram_singular_basis(matrix, side):
    """Exact singular values/basis through a symmetric Gram eigensystem."""
    gram = matrix.T @ matrix if side == "right" else matrix @ matrix.T
    gram = 0.5 * (gram + gram.T)
    scale = gram.diagonal().abs().mean().clamp_min(torch.finfo(gram.dtype).tiny)
    jitter = scale * 1e-7
    regularized = gram + jitter * torch.eye(
        len(gram), device=gram.device, dtype=gram.dtype
    )
    try:
        values, vectors = torch.linalg.eigh(regularized)
    except torch._C._LinAlgError:
        # cuSOLVER can reject highly rank-deficient Grams with large repeated
        # zero eigenspaces. CPU float64 LAPACK is used only for that numerical
        # fallback. The known diagonal jitter is removed from the eigenvalues.
        values, vectors = torch.linalg.eigh(regularized.double().cpu())
        values = values.to(device=matrix.device, dtype=matrix.dtype)
        vectors = vectors.to(device=matrix.device, dtype=matrix.dtype)
    order = torch.argsort(values, descending=True)
    return (values[order] - jitter).clamp_min(0).sqrt(), vectors[:, order]


def first_rank(rows,threshold):
    for rank,value in rows:
        if value>=threshold:return rank
    return None


def token_extremes(scores,tokenizer,count=5):
    high=torch.topk(scores,count).indices.tolist();low=torch.topk(-scores,count).indices.tolist();return {"positive":[tokenizer.decode([i]) for i in high],"negative":[tokenizer.decode([i]) for i in low]}


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text());arrays=np.load(SOURCE);basis=torch.from_numpy(arrays["basis"]).cuda().float();discovery_axes=torch.from_numpy(arrays["axes_direct_mlp"]).cuda().float();prototypes=torch.from_numpy(arrays["prototypes_direct_mlp"]).cuda().float();tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model("cuda");model.embedding.float().eval();model.blocks.float().eval();model.lm_head.float().eval();block=model.blocks[8];spec=config["prototype_discovery"]
    discovery_states,_=collect(model,8,spec,{**config,"positions":[config["causal_position"]]});discovery_features=discovery_states@basis;discovery_features=discovery_features/discovery_features.norm(dim=1,keepdim=True).clamp_min(1e-30);discovery_labels=grassmann_similarity(discovery_axes,prototypes).argmax(1);centroids=state_centroids(discovery_features,discovery_labels,config["clusters"])
    natural_states,_=collect(model,8,config["natural_states"],{**config,"positions":[config["causal_position"]]});features=natural_states@basis;features=features/features.norm(dim=1,keepdim=True).clamp_min(1e-30);labels=(features@centroids.T).argmax(1);row_ids=torch.arange(len(labels),device="cuda");train_mask=row_ids%2==0;test_mask=~train_mask;counts=[]
    for cluster in range(config["clusters"]):counts.append((cluster,int(((labels==cluster)&train_mask).sum()),int(((labels==cluster)&test_mask).sum())))
    eligible=[row for row in sorted(counts,key=lambda row:row[1],reverse=True) if row[1]>=config["minimum_train_sites"] and row[2]>=config["minimum_test_sites"]][:config["max_active_prototypes"]];active=[row[0] for row in eligible];results={};saved={"basis":basis.cpu().numpy(),"active":np.asarray(active)}
    down=block.mlp.Down.weight.float();left=block.mlp.Left.weight.float();right=block.mlp.Right.weight.float();embedding=model.embedding.weight.float()
    for cluster,train_count,test_count in eligible:
        print("restricted tensor",cluster,train_count,test_count,flush=True);edit=basis@prototypes[cluster];tensor=restricted_tensor(down,left,right,edit);train_x=natural_states[(labels==cluster)&train_mask];test_x=natural_states[(labels==cluster)&test_mask];square,inverse,covariance=ridge_covariance_transforms(train_x,config["covariance_floor_fraction"]);weighted=torch.einsum("oaj,jk->oak",tensor,square);context_matrix=weighted.permute(0,1,2).reshape(weighted.shape[0]*weighted.shape[1],weighted.shape[2]);context_singular,context_v=gram_singular_basis(context_matrix,"right");output_matrix=weighted.reshape(weighted.shape[0],-1);output_singular,output_u=gram_singular_basis(output_matrix,"left");whitened=test_x@inverse;target=torch.einsum("oaj,nj->nao",tensor,test_x);rank_rows=[]
        for rank in config["ranks"]:
            prediction=tucker_prediction(whitened,weighted,output_u,context_v,rank);rank_rows.append({"rank":rank,"heldout_r2":prediction_r2(target,prediction),"context_energy":energy_fraction(context_singular,rank),"output_energy":energy_fraction(output_singular,rank)})
        with torch.no_grad():native=block.mlp.Down((block.mlp.Left(edit.T)*block.mlp.Right(test_x[0])[None])+(block.mlp.Right(edit.T)*block.mlp.Left(test_x[0])[None]));formula=target[0];identity_relative_error=float((native-formula).norm()/native.norm().clamp_min(1e-30));context_read=inverse@context_v[:,0];context_scores=embedding@context_read;output_scores=model.lm_head(output_u[:,0][None]).squeeze(0)
        results[str(cluster)]={"train_sites":train_count,"test_sites":test_count,"identity_relative_error":identity_relative_error,"covariance":covariance,"smallest_rank_r2_90":first_rank([(row["rank"],row["heldout_r2"]) for row in rank_rows],.9),"rank_rows":rank_rows,"leading_context_tokens":token_extremes(context_scores,tokenizer),"leading_output_tokens":token_extremes(output_scores,tokenizer)};saved[f"output_basis_{cluster}"]=output_u[:,:max(config["ranks"])].cpu().numpy();saved[f"context_basis_{cluster}"]=context_v[:,:max(config["ranks"])].cpu().numpy()
    identity=max(row["identity_relative_error"] for row in results.values());b1=identity<=1e-5;b2=all(next(r for r in row["rank_rows"] if r["rank"]==64)["context_energy"]>=.9 for row in results.values());b3=all(next(r for r in row["rank_rows"] if r["rank"]==64)["output_energy"]>=.9 for row in results.values());b4=all(next(r for r in row["rank_rows"] if r["rank"]==64)["heldout_r2"]>=.8 for row in results.values());b5=all(next(r for r in row["rank_rows"] if r["rank"]==128)["heldout_r2"]>=.95 for row in results.values());ranks90=[row["smallest_rank_r2_90"] for row in results.values() if row["smallest_rank_r2_90"] is not None];b6=len(ranks90)==len(results) and float(np.median(ranks90))<=128
    hypotheses={"B1_restricted_tensor_identity_is_exact":{"supported":b1,"maximum_relative_error":identity},"B2_context_rank64_is_sufficient":{"supported":b2},"B3_output_rank64_is_sufficient":{"supported":b3},"B4_tucker64_predicts_heldout_writes":{"supported":b4},"B5_tucker128_is_high_fidelity":{"supported":b5},"B6_program_is_smaller_than_native_context":{"supported":b6}}
    np.savez_compressed(ARRAYS,**saved);OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"active_prototypes":active,"counts":counts,"results":results},indent=2,allow_nan=False));lines=["# Restricted tensor programs for Layer-8 sensitivity addresses","","For each frequently routed prototype, the exact bilinear weights are contracted with its four input directions, producing a tensor `output × edit × full context`. FineWeb route-conditioned covariance defines a natural context metric. A Tucker program jointly truncates context and output modes and predicts held-out native derivatives without fitting the target writes.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["",f"Active prototypes: `{active}`. Maximum exact-identity relative error: **{identity:.2e}**.","","![Restricted tensor program](layer8_restricted_tensor_program.png)","","| Prototype | Train/test sites | K90 | R²@64 / @128 | Context energy@64 | Output energy@64 |","|---:|---:|---:|---:|---:|---:|"]
    for cluster,row in results.items():
        r64=next(r for r in row["rank_rows"] if r["rank"]==64);r128=next(r for r in row["rank_rows"] if r["rank"]==128);lines.append(f"| {cluster} | {row['train_sites']}/{row['test_sites']} | {row['smallest_rank_r2_90']} | {r64['heldout_r2']:.3f}/{r128['heldout_r2']:.3f} | {r64['context_energy']:.3f} | {r64['output_energy']:.3f} |")
    lines += ["","The input prototype alone is an address. The restricted tensor plus current context is the exact conditional operation; Tucker rank measures whether that operation has a shorter reusable description."]
    REPORT.write_text("\n".join(lines)+"\n");figure,plots=plt.subplots(1,3,figsize=(15,4.2));ranks=config["ranks"]
    for cluster,row in results.items():plots[0].plot(ranks,[r["heldout_r2"] for r in row["rank_rows"]],marker="o",label=cluster);plots[1].plot(ranks,[r["context_energy"] for r in row["rank_rows"]],marker="o",label=cluster);plots[2].plot(ranks,[r["output_energy"] for r in row["rank_rows"]],marker="o",label=cluster)
    for plot,title in zip(plots,("Held-out derivative R²","Context-mode energy","Output-mode energy")):plot.set_xscale("log",base=2);plot.set_xticks(ranks,ranks);plot.set_title(title);plot.grid(alpha=.2)
    plots[0].legend(fontsize=7,ncol=2);figure.tight_layout();figure.savefig(FIGURE,dpi=180);plt.close(figure);print("wrote",OUTPUT,ARRAYS,REPORT,FIGURE,flush=True)


if __name__=="__main__":main()
