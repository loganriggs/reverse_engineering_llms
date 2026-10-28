#!/usr/bin/env python3
"""Exact-distance nonsemantic control for midpoint circuit compilation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_redteam import pair_bootstrap
from run_bilin18_layer01_prospective_tangent_router import (
    cosine, finite_example, jaccard, normalized_embedding, prefix_pre_from_x0,
    rms_jvp, score_rows, select_indices, tangent_product,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_midpoint_synthetic_control.json")
OUTPUT = TENSOR_ROOT / "layer01_midpoint_synthetic_control.json"
REPORT = TENSOR_ROOT / "LAYER01_MIDPOINT_SYNTHETIC_CONTROL.md"


def median(values): return float(np.median(np.asarray(values, dtype=np.float64)))


def sphere_endpoint(source, distance, seed):
    generator = torch.Generator(device=source.device).manual_seed(seed)
    random = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    unit = source / source.norm().clamp_min(1e-30)
    perpendicular = random - (random * unit).sum() * unit
    perpendicular = perpendicular / perpendicular.norm().clamp_min(1e-30)
    radius = source.norm()
    cosine_angle = (1 - distance.square() / (2 * radius.square())).clamp(-1, 1)
    sine_angle = torch.sqrt((1 - cosine_angle.square()).clamp_min(0))
    return radius * (cosine_angle * unit + sine_angle * perpendicular)


def direction_tangent_product(model, block, x0, direction, layer, final_position, path_point):
    def function(value): return prefix_pre_from_x0(model, value, layer, final_position)
    for prefix_block in model.blocks[:layer + 1]:
        prefix_block.attn.rotary.seq_len_cached = None
        prefix_block.attn.rotary.cos_cached = None
        prefix_block.attn.rotary.sin_cached = None
    with torch.enable_grad():
        pre, tangent = torch.autograd.functional.jvp(function, (x0 + path_point * direction,), (direction,), create_graph=False, strict=True)
    pre, tangent = pre[0].detach(), tangent[0].detach()
    state = F.rms_norm(pre, (model.config.n_embd,)); dstate = rms_jvp(pre, tangent)
    left, right = block.mlp.Left(state), block.mlp.Right(state)
    return F.linear(dstate, block.mlp.Left.weight.float()) * right + left * F.linear(dstate, block.mlp.Right.weight.float())


@torch.no_grad()
def synthetic_edit(model, ids, endpoint, layer, edit_position, final_position, rank):
    block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); x0=normalized_embedding(model,ids)
    direction=torch.zeros_like(x0); direction[:,edit_position]=endpoint-x0[:,edit_position]
    base=forward_layers(model,ids,last_layer=layer)[layer]
    counter_pre=prefix_pre_from_x0(model,x0+direction,layer,final_position)[0]
    base_state=F.rms_norm(base["pre"][0,final_position].float(),(model.config.n_embd,)); counter_state=F.rms_norm(counter_pre,(model.config.n_embd,))
    full=block.mlp.Left(counter_state)*block.mlp.Right(counter_state)-block.mlp.Left(base_state)*block.mlp.Right(base_state)
    midpoint=direction_tangent_product(model,block,x0,direction,layer,final_position,.5)
    actual_indices=select_indices(score_rows(full,decoder)["local"],rank); midpoint_indices=select_indices(score_rows(midpoint,decoder)["local"],rank)
    actual=torch.zeros_like(full); actual[actual_indices]=full[actual_indices]
    selected=torch.zeros_like(full); selected[midpoint_indices]=full[midpoint_indices]
    effects=downstream_fidelities(model,layer,base,{"full":full,"midpoint":selected,"oracle":actual},final_position)
    return {"midpoint":effects["midpoint"],"oracle":effects["oracle"],"support":jaccard(midpoint_indices,actual_indices),"cosine":cosine(midpoint,full)}


@torch.no_grad()
def semantic_edit(model,ids,target,layer,edit_position,final_position,rank):
    block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); base,_,full,_=finite_example(model,ids,target,layer,edit_position,final_position)
    midpoint,_=tangent_product(model,block,ids,target,layer,edit_position,final_position,path_point=.5)
    actual_indices=select_indices(score_rows(full,decoder)["local"],rank); midpoint_indices=select_indices(score_rows(midpoint,decoder)["local"],rank)
    actual=torch.zeros_like(full); actual[actual_indices]=full[actual_indices]
    selected=torch.zeros_like(full); selected[midpoint_indices]=full[midpoint_indices]
    effects=downstream_fidelities(model,layer,base,{"full":full,"midpoint":selected,"oracle":actual},final_position)
    return {"midpoint":effects["midpoint"],"oracle":effects["oracle"],"support":jaccard(midpoint_indices,actual_indices),"cosine":cosine(midpoint,full)}


@torch.no_grad()
def main():
    config=json.loads(CONFIG.read_text()); semantic_config=json.loads(Path(config["source_config"]).read_text()); midpoint_config=json.loads(Path(semantic_config["source_config"]).read_text()); redteam=json.loads(Path(midpoint_config["source_config"]).read_text()); source=json.loads(Path(redteam["source_config"]).read_text()); data=json.loads(Path(source["data_config"]).read_text()); pairs=json.loads(Path(source["evaluation_pair_config"]).read_text())
    edit_position,final_position=data["edit_position"],data["causal_position"]; model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    arrays={corpus:np.load(spec["path"],mmap_mode="r")[spec["start"]:spec["start"]+spec["rows"]] for corpus,spec in data["evaluation"].items()}; records={str(layer):[] for layer in source["layers"]}; max_mismatch=0.
    embeddings=F.rms_norm(model.embedding.weight.float(),(model.config.n_embd,))
    for corpus_index,(corpus,rows) in enumerate(arrays.items()):
        generator=np.random.default_rng(config["seed"]+corpus_index)
        for family,lexical_pairs in pairs["new_families"].items():
            for pair_index,(source_token,target_token) in enumerate(lexical_pairs):
                distance=(embeddings[source_token]-embeddings[target_token]).norm(); endpoints=[sphere_endpoint(embeddings[source_token],distance,config["seed"]+pair_index*100+index) for index in range(config["synthetic_directions_per_pair"])]
                for endpoint in endpoints: max_mismatch=max(max_mismatch,float(abs((endpoint-embeddings[source_token]).norm()-distance)/distance))
                windows=select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],generator)
                for context_index,(row_index,start) in enumerate(windows):
                    raw=np.asarray(rows[row_index,start:start+data["sequence_length"]],dtype=np.int64).copy(); ids=torch.as_tensor(raw,device="cuda")[None]
                    for layer in source["layers"]:
                        semantic=semantic_edit(model,ids,target_token,layer,edit_position,final_position,source["rank"])
                        controls=[synthetic_edit(model,ids,endpoint,layer,edit_position,final_position,source["rank"]) for endpoint in endpoints]
                        control={key:median([item[key] for item in controls]) for key in controls[0]}
                        records[str(layer)].append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","context":context_index,**{f"semantic_{key}":value for key,value in semantic.items()},**{f"synthetic_{key}":value for key,value in control.items()}})
                print("synthetic control",corpus,family,source_token,target_token,flush=True)
    summaries={}
    for layer in source["layers"]:
        summaries[str(layer)]={}
        for corpus_index,corpus in enumerate(arrays):
            selected=[row for row in records[str(layer)] if row["corpus"]==corpus]; difference,interval=pair_bootstrap(selected,"semantic_midpoint","synthetic_midpoint",config["bootstrap_samples"],config["seed"]+layer*100+corpus_index)
            summaries[str(layer)][corpus]={key:median([row[key] for row in selected]) for key in ("semantic_midpoint","semantic_oracle","semantic_support","semantic_cosine","synthetic_midpoint","synthetic_oracle","synthetic_support","synthetic_cosine")}; summaries[str(layer)][corpus].update({"semantic_minus_synthetic_pair_median":difference,"semantic_minus_synthetic_pair_ci":interval})
    row=lambda layer,corpus:summaries[str(layer)][corpus]
    hypotheses={
        "Y0_semantic_edits_compile_better_than_synthetic":{"supported":all(row(l,c)["semantic_midpoint"]-row(l,c)["synthetic_midpoint"]>=.05 for l in source["layers"] for c in arrays)},
        "Y1_semantic_edits_have_more_predictable_support":{"supported":all(row(l,c)["semantic_support"]-row(l,c)["synthetic_support"]>=.05 for l in source["layers"] for c in arrays)},
        "Y2_synthetic_edits_also_compile":{"supported":all(row(l,c)["synthetic_midpoint"]>=.8*row(l,c)["synthetic_oracle"] for l in source["layers"] for c in arrays)},
        "Y3_synthetic_direction_is_predictable":{"supported":all(row(l,c)["synthetic_cosine"]>=.95 for l in source["layers"] for c in arrays)},
        "Y4_synthetic_distance_is_exact":{"supported":max_mismatch<=1e-5,"maximum_relative_mismatch":max_mismatch},
        "Y5_semantic_fidelity_advantage_survives_pair_bootstrap":{"supported":all(row(l,c)["semantic_minus_synthetic_pair_ci"][0]>0 for l in source["layers"] for c in arrays)},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_distance_mismatch":max_mismatch,"summaries":summaries,"records":records},indent=2))
    lines=["# Exact-distance synthetic control for midpoint compilation","","For every semantic source→target edit, two random endpoints are constructed on the normalized embedding sphere at exactly the same chord distance from the source. They cannot encode the declared lexical relation. Semantic and synthetic edits use the same source contexts, rank, midpoint-JVP budget, finite-delta patch, and output metric.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Semantic midpoint/oracle/support/cosine | Synthetic midpoint/oracle/support/cosine | Semantic-synthetic pair median (95% CI) |","|---:|---|---:|---:|---:|"]
    for l in source["layers"]:
        for c in arrays:
            item=row(l,c); ci=item["semantic_minus_synthetic_pair_ci"]; lines.append(f"| {l} | {c} | {item['semantic_midpoint']:.3f}/{item['semantic_oracle']:.3f}/{item['semantic_support']:.3f}/{item['semantic_cosine']:.3f} | {item['synthetic_midpoint']:.3f}/{item['synthetic_oracle']:.3f}/{item['synthetic_support']:.3f}/{item['synthetic_cosine']:.3f} | {item['semantic_minus_synthetic_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) |")
    lines += ["",f"Maximum relative edit-distance mismatch: **{max_mismatch:.2e}**.","","Success on synthetic endpoints would identify generic low-degree tensor geometry, not semantic abstraction. A semantic advantage in support but not fidelity would instead mean natural edits are more natively axis-aligned without being easier to reproduce behaviorally."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
