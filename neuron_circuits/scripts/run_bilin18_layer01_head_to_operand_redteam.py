#!/usr/bin/env python3
"""Double-precision and Shapley red-team of head-to-operand routes."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_dynamic_neuron_roles import role_statistics
from run_bilin18_layer01_final_attention_routes import attention_factorization
from run_bilin18_layer01_final_bilinear_readout import bilinear_terms
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_head_to_operand_redteam.json")
OUTPUT=TENSOR_ROOT/"layer01_head_to_operand_redteam.json"
REPORT=TENSOR_ROOT/"LAYER01_HEAD_TO_OPERAND_REDTEAM.md"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def factor_values(block,state):
    normalized=F.rms_norm(state,(state.shape[-1],)); return F.linear(normalized,block.mlp.Left.weight.double()),F.linear(normalized,block.mlp.Right.weight.double())


@torch.no_grad()
def straight_sources(block,base,components,points):
    directions=torch.stack(list(components.values())).double(); total=directions.sum(0); nodes,weights=np.polynomial.legendre.leggauss(points); left=torch.zeros((len(directions),block.mlp.Left.out_features),device=base.device,dtype=torch.float64); right=torch.zeros_like(left); lw=block.mlp.Left.weight.double(); rw=block.mlp.Right.weight.double()
    for raw_node,raw_weight in zip(nodes,weights):
        node=(float(raw_node)+1)/2; weight=float(raw_weight)/2; state=base.double()+node*total; epsilon=torch.finfo(state.dtype).eps; inverse=torch.rsqrt(state.square().mean()+epsilon); tangent=directions*inverse-state*(directions*state).mean(-1,keepdim=True)*inverse.pow(3); left+=weight*F.linear(tangent,lw); right+=weight*F.linear(tangent,rw)
    start_left,start_right=factor_values(block,base.double()); end_left,end_right=factor_values(block,base.double()+total); error=max(float((left.sum(0)-(end_left-start_left)).norm()/(end_left-start_left).norm().clamp_min(1e-30)),float((right.sum(0)-(end_right-start_right)).norm()/(end_right-start_right).norm().clamp_min(1e-30)))
    return left,right,start_left,start_right,end_left,end_right,error


@torch.no_grad()
def permutation_shapley(block,base,components,permutations,seed):
    names=list(components); directions=torch.stack(list(components.values())).double(); generator=np.random.default_rng(seed); left=torch.zeros((len(names),block.mlp.Left.out_features),device=base.device,dtype=torch.float64); right=torch.zeros_like(left)
    for _ in range(permutations):
        order=generator.permutation(len(names)); state=base.double().clone(); previous_left,previous_right=factor_values(block,state)
        for index in order:
            state=state+directions[index]; current_left,current_right=factor_values(block,state); left[index]+=current_left-previous_left; right[index]+=current_right-previous_right; previous_left,previous_right=current_left,current_right
    left/=permutations; right/=permutations; start_left,start_right=factor_values(block,base.double()); end_left,end_right=factor_values(block,base.double()+directions.sum(0)); error=max(float((left.sum(0)-(end_left-start_left)).norm()/(end_left-start_left).norm().clamp_min(1e-30)),float((right.sum(0)-(end_right-start_right)).norm()/(end_right-start_right).norm().clamp_min(1e-30)))
    return left,right,error


def attribution_metrics(values,indices,frozen_indices,names):
    absolute=values[:,indices].abs(); denominator=absolute.sum(0).clamp_min(1e-30); shares=absolute/denominator; dominant=shares.argmax(0); frozen=shares[frozen_indices].sum(0); counts=collections.Counter(names[index] for index in dominant.tolist())
    return dominant,float(frozen.median()),counts


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); source_config=json.loads(Path(config["source_config"]).read_text()); roles=json.loads(Path(source_config["source_config"]).read_text()); bilinear=json.loads(Path(roles["source_config"]).read_text()); final_config=json.loads(Path(bilinear["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source); new_config=json.loads(Path(config["new_pair_config"]).read_text()); frozen=json.loads(Path(config["frozen_head_results"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; results={}; max_straight_error=0.; max_shapley_error=0.
    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.double(); frozen_heads=frozen["layers"][str(layer)]["banks"]["global"]; records=[]
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(source_config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for pair_index,(source_token,target_token) in enumerate(pairs):
                    windows=select_windows(rows,source_token,edit_position,final_position,source_config["contexts_per_pair"],rng)[:config["contexts_per_pair"]]
                    for context_index,(row_index,start) in enumerate(windows):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token; base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; _,_,head_writes,_,_,_=attention_factorization(block,base,counter,final_position); components={}
                        if layer>0: components["carried"]=counter["mixed"][0,final_position].float()-base["mixed"][0,final_position].float()
                        components.update({f"head{head}":head_writes[head] for head in range(block.attn.n_head)}); names=list(components); straight_left,straight_right,base_left,base_right,end_left,end_right,error=straight_sources(block,base["pre"][0,final_position].float(),components,config["quadrature_points"]); shapley_left,shapley_right,shapley_error=permutation_shapley(block,base["pre"][0,final_position].float(),components,config["shapley_permutations"],config["seed"]+layer*10000+corpus_index*1000+pair_index); max_straight_error=max(max_straight_error,error); max_shapley_error=max(max_shapley_error,shapley_error); terms={"left_cross":(end_left-base_left)*base_right,"right_cross":base_left*(end_right-base_right),"quadratic":(end_left-base_left)*(end_right-base_right)}; full=sum(terms.values()); target=F.linear(full,decoder); contribution=full*(target@decoder)/target.square().sum().clamp_min(1e-30); indices=contribution.abs().topk(config["rank"]).indices; _,side,_,_=role_statistics(terms["left_cross"],terms["right_cross"],terms["quadratic"]); straight_values=torch.where(side[None],straight_left,straight_right); shapley_values=torch.where(side[None],shapley_left,shapley_right); frozen_indices=[names.index(f"head{head}") for head in frozen_heads]; straight_dom,straight_frozen,straight_counts=attribution_metrics(straight_values,indices,frozen_indices,names); shapley_dom,shapley_frozen,shapley_counts=attribution_metrics(shapley_values,indices,frozen_indices,names); records.append({"corpus":corpus,"family":family_name,"pair":f"{source_token}->{target_token}","dominant_agreement":float((straight_dom==shapley_dom).float().mean()),"straight_frozen":straight_frozen,"shapley_frozen":shapley_frozen,"straight_top_source":straight_counts.most_common(1)[0][0],"shapley_top_source":shapley_counts.most_common(1)[0][0]})
                    print("operand redteam",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={corpus:{"dominant_agreement":median([row["dominant_agreement"] for row in records if row["corpus"]==corpus]),"straight_frozen":median([row["straight_frozen"] for row in records if row["corpus"]==corpus]),"shapley_frozen":median([row["shapley_frozen"] for row in records if row["corpus"]==corpus]),"straight_top_source":collections.Counter(row["straight_top_source"] for row in records if row["corpus"]==corpus).most_common(1)[0],"shapley_top_source":collections.Counter(row["shapley_top_source"] for row in records if row["corpus"]==corpus).most_common(1)[0]} for corpus in fixed["evaluation"]}; results[str(layer)]={"summaries":summaries,"records":records,"frozen_heads":frozen_heads}
    def row(layer,corpus): return results[str(layer)]["summaries"][corpus]
    hypotheses={
        "P0_double_quadrature_resolves_numeric_error":{"supported":max_straight_error<=2e-5,"maximum_relative_error":max_straight_error},
        "P1_shapley_sources_recompose":{"supported":max_shapley_error<=2e-5,"maximum_relative_error":max_shapley_error},
        "P2_dominant_source_is_attribution_robust":{"supported":all(row(0,c)["dominant_agreement"]>=.75 and row(1,c)["dominant_agreement"]>=.6 for c in fixed["evaluation"])},
        "P3_frozen_head_coverage_is_attribution_robust":{"supported":all(abs(row(l,c)["straight_frozen"]-row(l,c)["shapley_frozen"])<=.1 for l in config["layers"] for c in fixed["evaluation"])},
        "P4_layer0_has_more_fixed_transport_than_layer1":{"supported":all(row(0,c)[method]>=row(1,c)[method]+.15 for c in fixed["evaluation"] for method in ("straight_frozen","shapley_frozen"))},
        "P5_layer0_head7_is_a_stable_anchor":{"supported":all(row(0,c)[method][0]=="head7" for c in fixed["evaluation"] for method in ("straight_top_source","shapley_top_source"))},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_straight_error":max_straight_error,"maximum_shapley_error":max_shapley_error,"layers":results},indent=2)); lines=["# Numeric and attribution-path red-team of head-to-operand routes","","The original straight-path result is recomputed with float64 32-point quadrature. A symmetric 32-permutation Monte Carlo Shapley allocation supplies a different finite attribution across head/carried sources. Both operate on one preregistered context for every prospective lexical pair and corpus.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Dominant-source agreement | Frozen share straight / Shapley | Most common source straight / Shapley |","|---:|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); lines.append(f"| {layer} | {corpus} | {item['dominant_agreement']:.3f} | {item['straight_frozen']:.3f} / {item['shapley_frozen']:.3f} | {item['straight_top_source'][0]} ({item['straight_top_source'][1]}) / {item['shapley_top_source'][0]} ({item['shapley_top_source'][1]}) |")
    lines += ["",f"Maximum float64 straight-path recomposition error: **{max_straight_error:.2e}**. Maximum Shapley recomposition error: **{max_shapley_error:.2e}**.","","The locked float32 eight-point failure remains part of the primary report. This red-team distinguishes numerical integration error from source-attribution path dependence; it does not retroactively change the original gate."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
