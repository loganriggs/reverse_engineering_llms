#!/usr/bin/env python3
"""Prospective finite-edit compiler and path-degree sweep through Layers 0--4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_edit_path_degree import polynomial_metrics
from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_router import (
    cosine, jaccard, normalized_embedding, rms_jvp, score_rows, select_indices,
)
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer04_path_compiler.json")
OUTPUT = TENSOR_ROOT / "layer04_path_compiler.json"
REPORT = TENSOR_ROOT / "LAYER04_PATH_COMPILER.md"


def median(values): return float(np.median(np.asarray(values, dtype=np.float64)))


def clear_rotary(model, last_layer):
    for block in model.blocks[:last_layer + 1]:
        block.attn.rotary.seq_len_cached = None
        block.attn.rotary.cos_cached = None
        block.attn.rotary.sin_cached = None


def prefix_pres(model, x0, last_layer, position):
    """Return every pre-MLP state at one position in one shared prefix pass."""
    x, value_bus, rows = x0, None, []
    for block in model.blocks[:last_layer + 1]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (model.config.n_embd,)), value_bus)
        pre = mixed + attention
        rows.append(pre[:, position])
        x = pre + block.mlp(F.rms_norm(pre, (model.config.n_embd,)))
    return torch.stack(rows, dim=1)


def shared_jvp(model, x0, direction, last_layer, position, path_point):
    clear_rotary(model, last_layer)
    with torch.enable_grad():
        output, tangent = torch.autograd.functional.jvp(
            lambda value: prefix_pres(model, value, last_layer, position),
            (x0 + path_point * direction,), (direction,), create_graph=False, strict=True,
        )
    return output[0].detach(), tangent[0].detach()


def product_from_pre(block, pre, dimension):
    state = F.rms_norm(pre, (dimension,))
    return block.mlp.Left(state) * block.mlp.Right(state)


def product_tangent(block, pre, tangent, dimension):
    state, dstate = F.rms_norm(pre, (dimension,)), rms_jvp(pre, tangent)
    left, right = block.mlp.Left(state), block.mlp.Right(state)
    return (
        F.linear(dstate, block.mlp.Left.weight.to(dstate.dtype)) * right
        + left * F.linear(dstate, block.mlp.Right.weight.to(dstate.dtype))
    )


@torch.no_grad()
def main(config_path=CONFIG, output_path=OUTPUT, report_path=REPORT):
    config=json.loads(Path(config_path).read_text()); data=json.loads(Path(config["data_config"]).read_text()); pair_config=json.loads(Path(config["pair_config"]).read_text()); edit_position,final_position=data["edit_position"],data["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays={corpus:np.load(spec["path"],mmap_mode="r")[spec["start"]:spec["start"]+spec["rows"]] for corpus,spec in data["evaluation"].items()}; last_layer=max(config["layers"]); records={str(layer):[] for layer in config["layers"]}
    path_nodes=config["train_nodes"]+config["evaluation_nodes"]
    for corpus_index,(corpus,rows) in enumerate(arrays.items()):
        generator=np.random.default_rng(config["seed"]+corpus_index)
        for family,pairs in pair_config["new_families"].items():
            for source_token,target_token in pairs:
                windows=select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],generator)
                for context_index,(row_index,start) in enumerate(windows):
                    raw=np.asarray(rows[row_index,start:start+data["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token; ids=torch.as_tensor(raw,device="cuda")[None]; changed_ids=torch.as_tensor(changed,device="cuda")[None]
                    base_records=forward_layers(model,ids,last_layer=last_layer); counter_records=forward_layers(model,changed_ids,last_layer=last_layer)
                    x0=normalized_embedding(model,ids); target=normalized_embedding(model,torch.tensor([target_token],device="cuda"))[0]; direction=torch.zeros_like(x0); direction[:,edit_position]=target-x0[:,edit_position]
                    base_pres,base_tangents=shared_jvp(model,x0,direction,last_layer,final_position,0.0); midpoint_pres,midpoint_tangents=shared_jvp(model,x0,direction,last_layer,final_position,config["midpoint"])
                    path_pres=[]
                    for node in path_nodes+[config["small_dose"]]:
                        clear_rotary(model,last_layer); path_pres.append(prefix_pres(model,x0+node*direction,last_layer,final_position)[0].detach())
                    for layer in config["layers"]:
                        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); dimension=model.config.n_embd
                        base_state=F.rms_norm(base_records[layer]["pre"][0,final_position].float(),(dimension,)); counter_state=F.rms_norm(counter_records[layer]["pre"][0,final_position].float(),(dimension,)); full=block.mlp.Left(counter_state)*block.mlp.Right(counter_state)-block.mlp.Left(base_state)*block.mlp.Right(base_state)
                        base_prediction=product_tangent(block,base_pres[layer],base_tangents[layer],dimension); midpoint_prediction=product_tangent(block,midpoint_pres[layer],midpoint_tangents[layer],dimension)
                        products=torch.stack([product_from_pre(block,pres[layer],dimension) for pres in path_pres]); path_delta_products=products[:-1]; small_delta=products[-1]-products[0]
                        actual_indices=select_indices(score_rows(full,decoder)["local"],config["rank"]); midpoint_indices=select_indices(score_rows(midpoint_prediction,decoder)["local"],config["rank"])
                        actual=torch.zeros_like(full); actual[actual_indices]=full[actual_indices]; midpoint_selected=torch.zeros_like(full); midpoint_selected[midpoint_indices]=full[midpoint_indices]
                        effects=downstream_fidelities(model,layer,base_records[layer],{"full":full,"midpoint":midpoint_selected,"oracle":actual},final_position)
                        count=len(config["train_nodes"]); path_metrics=polynomial_metrics(path_delta_products[:count],path_delta_products[count:],config["train_nodes"],config["evaluation_nodes"],config["polynomial_degrees"])
                        records[str(layer)].append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","context":context_index,"small_cosine":cosine(base_prediction,small_delta),"base_cosine":cosine(base_prediction,full),"midpoint_cosine":cosine(midpoint_prediction,full),"midpoint_norm_error":float((midpoint_prediction-full).norm()/full.norm().clamp_min(1e-30)),"support_jaccard":jaccard(midpoint_indices,actual_indices),"midpoint_fidelity":effects["midpoint"],"oracle_fidelity":effects["oracle"],"path_metrics":path_metrics})
                print("layer04 compiler",corpus,family,source_token,target_token,flush=True)
    summaries={}
    for layer in config["layers"]:
        summaries[str(layer)]={}
        for corpus in arrays:
            selected=[row for row in records[str(layer)] if row["corpus"]==corpus]
            summaries[str(layer)][corpus]={key:median([row[key] for row in selected]) for key in ("small_cosine","base_cosine","midpoint_cosine","midpoint_norm_error","support_jaccard","midpoint_fidelity","oracle_fidelity")}
            summaries[str(layer)][corpus]["causal_retention"]=summaries[str(layer)][corpus]["midpoint_fidelity"]/max(1e-30,summaries[str(layer)][corpus]["oracle_fidelity"])
            summaries[str(layer)][corpus]["path_metrics"]={key:median([row["path_metrics"][key] for row in selected]) for key in [str(degree) for degree in config["polynomial_degrees"]]+["quadratic_share","rank2_energy"]}
    row=lambda layer,corpus:summaries[str(layer)][corpus]
    if last_layer <= 4:
        breaks = None
        hypotheses={
            "D0_small_edit_tangent_is_valid":{"supported":all(row(l,c)["small_cosine"]>=.98 for l in config["layers"] for c in arrays)},
            "D1_midpoint_direction_survives_to_layer4":{"supported":all(row(l,c)["midpoint_cosine"]>=.90 for l in config["layers"] for c in arrays)},
            "D2_midpoint_support_survives_to_layer4":{"supported":all(row(l,c)["support_jaccard"]>=.50 for l in config["layers"] for c in arrays)},
            "D3_midpoint_causal_retention_survives_to_layer4":{"supported":all(row(l,c)["causal_retention"]>=.80 for l in config["layers"] for c in arrays)},
            "D4_quadratic_path_survives_to_layer4":{"supported":all(row(l,c)["path_metrics"]["2"]>=.98 for l in config["layers"] for c in arrays)},
            "D5_path_remains_nearly_planar":{"supported":all(row(l,c)["path_metrics"]["rank2_energy"]>=.98 for l in config["layers"] for c in arrays)},
            "D6_depth_introduces_measurable_path_complexity":{"supported":all((row(0,c)["midpoint_cosine"]-row(4,c)["midpoint_cosine"]>=.05) or (row(0,c)["path_metrics"]["rank2_energy"]-row(4,c)["path_metrics"]["rank2_energy"]>=.01) for c in arrays)},
            "D7_layer01_midpoint_result_replicates":{"supported":all(row(l,c)["midpoint_cosine"]>=.95 and row(l,c)["support_jaccard"]>=.65 and row(l,c)["causal_retention"]>=.80 for l in (0,1) for c in arrays)},
        }
        title = "# Prospective finite-edit path compiler through Layers 0--4"
    else:
        thresholds = config["break_thresholds"]
        checks = {
            "midpoint_cosine": lambda item: item["midpoint_cosine"] < thresholds["midpoint_cosine"],
            "support_jaccard": lambda item: item["support_jaccard"] < thresholds["support_jaccard"],
            "midpoint_fidelity": lambda item: item["midpoint_fidelity"] < thresholds["midpoint_fidelity"],
            "quadratic_path_fidelity": lambda item: item["path_metrics"]["2"] < thresholds["quadratic_path_fidelity"],
            "rank2_path_energy": lambda item: item["path_metrics"]["rank2_energy"] < thresholds["rank2_path_energy"],
        }
        breaks = {
            corpus: {
                name: next((layer for layer in config["layers"] if check(row(layer, corpus))), None)
                for name, check in checks.items()
            }
            for corpus in arrays
        }
        hypotheses={
            "D0_small_edit_tangent_is_valid":{"supported":all(row(l,c)["small_cosine"]>=.98 for l in config["layers"] for c in arrays)},
            "D1_midpoint_direction_survives_full_depth":{"supported":all(row(l,c)["midpoint_cosine"]>=thresholds["midpoint_cosine"] for l in config["layers"] for c in arrays)},
            "D2_midpoint_support_survives_full_depth":{"supported":all(row(l,c)["support_jaccard"]>=thresholds["support_jaccard"] for l in config["layers"] for c in arrays)},
            "D3_midpoint_causal_fidelity_survives_full_depth":{"supported":all(row(l,c)["midpoint_fidelity"]>=thresholds["midpoint_fidelity"] and row(l,c)["midpoint_fidelity"]>=row(l,c)["oracle_fidelity"]-.15 for l in config["layers"] for c in arrays)},
            "D4_quadratic_path_survives_full_depth":{"supported":all(row(l,c)["path_metrics"]["2"]>=thresholds["quadratic_path_fidelity"] for l in config["layers"] for c in arrays)},
            "D5_path_remains_nearly_planar_full_depth":{"supported":all(row(l,c)["path_metrics"]["rank2_energy"]>=thresholds["rank2_path_energy"] for l in config["layers"] for c in arrays)},
            "D6_a_path_compiler_break_exists_after_layer4":{"supported":all(any(value is not None and value>4 for value in breaks[corpus].values()) for corpus in arrays)},
            "D7_layer04_result_replicates":{"supported":all(row(l,c)["midpoint_cosine"]>=.90 and row(l,c)["support_jaccard"]>=.50 and row(l,c)["path_metrics"]["2"]>=.98 and row(l,c)["path_metrics"]["rank2_energy"]>=.98 for l in range(5) for c in arrays)},
        }
        title = "# Prospective finite-edit path compiler through all 18 layers"
    Path(output_path).write_text(json.dumps({"config":config,"hypotheses":hypotheses,"breaks":breaks,"summaries":summaries,"records":records},indent=2))
    lines=[title,"",f"One shared checkpoint-native prefix JVP produces pre-MLP tangents for all {len(config['layers'])} layers on the same examples. Midpoint-selected actual channel deltas and realized-local controls use the same rank 256 and are propagated through each complete suffix. Five fit nodes and four interleaved held-out nodes measure per-example product-path degree.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    if breaks is not None:
        lines += ["", "First registered threshold failure by corpus: `" + json.dumps(breaks, sort_keys=True) + "`."]
    lines += ["","| Layer | Corpus | Small/base/mid cosine | Mid norm error | Support Jaccard | Mid/realized-local fidelity (ratio) | Path d1/d2/d3 | Rank-2 energy |","|---:|---|---:|---:|---:|---:|---:|---:|"]
    for l in config["layers"]:
        for c in arrays:
            item=row(l,c); path=item["path_metrics"]; lines.append(f"| {l} | {c} | {item['small_cosine']:.3f}/{item['base_cosine']:.3f}/{item['midpoint_cosine']:.3f} | {item['midpoint_norm_error']:.3f} | {item['support_jaccard']:.3f} | {item['midpoint_fidelity']:.3f}/{item['oracle_fidelity']:.3f} ({item['causal_retention']:.3f}) | {path['1']:.3f}/{path['2']:.3f}/{path['3']:.3f} | {path['rank2_energy']:.4f} |")
    lines += ["","The sweep locates a depth transition only if direction, support, causal retention, or held-out path geometry degrades under an unchanged assay. Passing through Layer 4 would mean the low-degree conditional-program ontology extends beyond the directly lexical layers."]
    Path(report_path).write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(report_path)


if __name__=="__main__": main()
