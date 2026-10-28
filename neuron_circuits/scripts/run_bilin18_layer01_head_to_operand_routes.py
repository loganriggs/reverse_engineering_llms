#!/usr/bin/env python3
"""Map exact attention-head/carried sources into Layer-0/1 bilinear operands."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_dynamic_neuron_roles import role_statistics
from run_bilin18_layer01_final_attention_routes import attention_factorization
from run_bilin18_layer01_final_bilinear_readout import bilinear_terms
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from transformers import AutoTokenizer


CONFIG=Path("configs/bilin18_layer01_head_to_operand_routes.json")
OUTPUT=TENSOR_ROOT/"layer01_head_to_operand_routes.json"
REPORT=TENSOR_ROOT/"LAYER01_HEAD_TO_OPERAND_ROUTES.md"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def rmsnorm_jvp(state,directions):
    epsilon=torch.finfo(state.dtype).eps; inverse=torch.rsqrt(state.square().mean()+epsilon); covariance=(directions*state).mean(-1,keepdim=True)
    return directions*inverse-state*covariance*inverse.pow(3)


def integrated_factor_sources(block,base_pre,components,points):
    names=list(components); directions=torch.stack(list(components.values())); total=directions.sum(0); nodes,weights=np.polynomial.legendre.leggauss(points); nodes=(nodes+1)/2; weights=weights/2; left=torch.zeros((len(names),block.mlp.Left.out_features),device=base_pre.device); right=torch.zeros_like(left)
    for node,weight in zip(nodes,weights):
        state=base_pre+float(node)*total; tangent=rmsnorm_jvp(state,directions); left+=float(weight)*F.linear(tangent,block.mlp.Left.weight.float()); right+=float(weight)*F.linear(tangent,block.mlp.Right.weight.float())
    base_state=F.rms_norm(base_pre,(base_pre.shape[-1],)); end_state=F.rms_norm(base_pre+total,(base_pre.shape[-1],)); exact_left=block.mlp.Left(end_state)-block.mlp.Left(base_state); exact_right=block.mlp.Right(end_state)-block.mlp.Right(base_state); left_error=float((left.sum(0)-exact_left).norm()/exact_left.norm().clamp_min(1e-30)); right_error=float((right.sum(0)-exact_right).norm()/exact_right.norm().clamp_min(1e-30))
    return names,left,right,base_state,end_state,max(left_error,right_error)


def source_metrics(source_values,indices,frozen_source_indices):
    values=source_values[:,indices].abs(); denominator=values.sum(0).clamp_min(1e-30); sorted_values=values.sort(dim=0,descending=True).values; top3=(sorted_values[:min(3,len(values))].sum(0)/denominator); frozen=(values[frozen_source_indices].sum(0)/denominator) if frozen_source_indices else torch.zeros_like(top3); dominant=values.argmax(0)
    return top3,frozen,dominant,values/denominator


def aggregate_pair_sources(rows,rank):
    score=collections.defaultdict(float); votes=collections.defaultdict(lambda:collections.defaultdict(float))
    for row in rows:
        for channel,value,source in zip(row["indices"],row["scores"],row["dominant_sources"]): score[channel]+=value; votes[channel][source]+=value
    selected=sorted(score,key=score.get,reverse=True)[:rank]
    return {channel:max(votes[channel],key=votes[channel].get) for channel in selected}


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); roles=json.loads(Path(config["source_config"]).read_text()); bilinear=json.loads(Path(roles["source_config"]).read_text()); final_config=json.loads(Path(bilinear["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source); new_config=json.loads(Path(config["new_pair_config"]).read_text()); frozen=json.loads(Path(config["frozen_head_results"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True); results={}; maximum_integral_error=0.; maximum_native_error=0.
    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); frozen_heads=frozen["layers"][str(layer)]["banks"]["global"]; records=[]; concrete=[]; edge_totals=collections.defaultdict(float); random_indices=torch.randperm(decoder.shape[1],generator=torch.Generator(device="cpu").manual_seed(config["seed"]+100+layer))[:config["rank"]].cuda()
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    pair=f"{source_token}->{target_token}"
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token; base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; _,_,head_writes,native_attention,algebraic_error,native_error=attention_factorization(block,base,counter,final_position); maximum_native_error=max(maximum_native_error,native_error,algebraic_error); components={}
                        if layer>0: components["carried"]=counter["mixed"][0,final_position].float()-base["mixed"][0,final_position].float()
                        components.update({f"head{head}":head_writes[head] for head in range(block.attn.n_head)}); names,left_sources,right_sources,base_state,end_state,error=integrated_factor_sources(block,base["pre"][0,final_position].float(),components,config["quadrature_points"]); maximum_integral_error=max(maximum_integral_error,error); native_end=F.rms_norm(counter["pre"][0,final_position].float(),(model.config.n_embd,)); maximum_native_error=max(maximum_native_error,float((end_state-native_end).norm()/native_end.norm().clamp_min(1e-30))); terms=bilinear_terms(block,base_state,end_state); full=sum(terms.values()); target_write=F.linear(full,decoder); contribution=full*(target_write@decoder)/target_write.square().sum().clamp_min(1e-30); indices=contribution.abs().topk(config["rank"]).indices; _,side,_,_=role_statistics(terms["left_cross"],terms["right_cross"],terms["quadratic"]); source_values=torch.where(side[None],left_sources,right_sources); frozen_source_indices=[names.index(f"head{head}") for head in frozen_heads]; top3,frozen_share,dominant,shares=source_metrics(source_values,indices,frozen_source_indices); random_top3,_,_,_=source_metrics(source_values,random_indices,frozen_source_indices); carried_share=shares[names.index("carried")] if "carried" in names else torch.zeros_like(top3); cancellation=source_values[:,indices].abs().sum(0)/source_values[:,indices].sum(0).abs().clamp_min(1e-30)
                        for col,channel in enumerate(indices.tolist()):
                            for source_index,name in enumerate(names): edge_totals[(name,channel)]+=float(contribution[channel].abs()*shares[source_index,col])
                        record={"corpus":corpus,"family":family_name,"pair":pair,"context":context_index,"indices":indices.cpu().tolist(),"scores":contribution[indices].abs().cpu().tolist(),"dominant_sources":[names[index] for index in dominant.cpu().tolist()],"top3_share":float(top3.median()),"frozen_share":float(frozen_share.median()),"carried_share":float(carried_share.median()),"cancellation_ratio":float(cancellation.median()),"random_top3_share":float(random_top3.median())}; records.append(record)
                        if corpus=="fineweb" and context_index==0 and pair in config["concrete_pairs"]:
                            concrete.append({"pair":pair,"tokens":f"{tokenizer.decode([source_token])}->{tokenizer.decode([target_token])}","context":tokenizer.decode(raw[max(0,edit_position-10):final_position+1]),"channels":[{"channel":int(channel),"projection":float(contribution[channel]),"operand_side":"left" if bool(side[channel]) else "right","dominant_source":names[int(dominant[col])],"source_share":float(shares[int(dominant[col]),col]),"top3_source_share":float(top3[col])} for col,channel in enumerate(indices[:config["concrete_channels"]].tolist())]})
                    print("head operand",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={}; recurring={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]; summaries[corpus]={key:median([row[key] for row in selected]) for key in ("top3_share","frozen_share","carried_share","cancellation_ratio","random_top3_share")}
            by_channel=collections.defaultdict(list)
            for row in selected:
                for channel,source_name in zip(row["indices"],row["dominant_sources"]): by_channel[channel].append(source_name)
            agreements=[]
            for values in by_channel.values():
                if len(values)>=config["recurrence_minimum"]: agreements.append(max(collections.Counter(values).values())/len(values))
            recurring[corpus]={"channel_count":len(agreements),"median_source_agreement":median(agreements) if agreements else None}
        pair_rows=[]
        for pair in sorted({row["pair"] for row in records}):
            left=aggregate_pair_sources([row for row in records if row["corpus"]=="fineweb" and row["pair"]==pair],config["rank"]); right=aggregate_pair_sources([row for row in records if row["corpus"]=="tinystories" and row["pair"]==pair],config["rank"]); common=set(left)&set(right); pair_rows.append({"pair":pair,"intersection":len(common),"source_agreement":float(np.mean([left[channel]==right[channel] for channel in common])) if common else None})
        top_edges=[{"source":name,"channel":channel,"weight":weight} for (name,channel),weight in sorted(edge_totals.items(),key=lambda item:item[1],reverse=True)[:20]]; results[str(layer)]={"source_names":names,"frozen_heads":frozen_heads,"summaries":summaries,"recurring":recurring,"cross_corpus_source_agreement":median([row["source_agreement"] for row in pair_rows if row["source_agreement"] is not None]),"pair_rows":pair_rows,"top_edges":top_edges,"concrete":concrete,"records":records}
    hypotheses={
        "M0_integrated_sources_recompose_factors":{"supported":maximum_integral_error<=2e-5,"maximum_relative_error":maximum_integral_error},
        "M1_three_sources_explain_each_operand":{"supported":all(results[str(l)]["summaries"][c]["top3_share"]>=.7 for l in config["layers"] for c in fixed["evaluation"])},
        "M2_frozen_transport_heads_cover_operands":{"supported":all(results[str(l)]["summaries"][c]["frozen_share"]>=.5 for l in config["layers"] for c in fixed["evaluation"])},
        "M3_layer0_source_roles_are_more_recurrent":{"supported":all(results["0"]["recurring"][c]["median_source_agreement"]>=results["1"]["recurring"][c]["median_source_agreement"]+.1 for c in fixed["evaluation"])},
        "M4_layer0_source_roles_transfer_ood":{"supported":results["0"]["cross_corpus_source_agreement"]>=.6 and results["1"]["cross_corpus_source_agreement"]<.6},
        "M5_layer1_carried_state_is_material":{"supported":all(results["1"]["summaries"][c]["carried_share"]>=.15 for c in fixed["evaluation"])},
        "M6_causal_selection_enriches_source_concentration":{"supported":all(results[str(l)]["summaries"][c]["top3_share"]-results[str(l)]["summaries"][c]["random_top3_share"]>=.05 for l in config["layers"] for c in fixed["evaluation"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_integral_error":maximum_integral_error,"maximum_native_order_error":maximum_native_error,"layers":results},indent=2)); lines=["# Attention-head and carried-state routes into bilinear operands","","Each native head's exact final-position attention delta, plus Layer-1 carried-state change, is propagated through the analytic RMSNorm Jacobian and integrated along the straight finite edit path. The resulting source effects recompose every left/right factor endpoint. For each causally selected neuron, attribution follows whichever factor supplies its dominant cross term.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Top-3 source share | Frozen-head share | Carried share | Cancellation ratio | Random top-3 | Recurring source agreement |","|---:|---|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=results[str(layer)]["summaries"][corpus]; rec=results[str(layer)]["recurring"][corpus]; lines.append(f"| {layer} | {corpus} | {item['top3_share']:.3f} | {item['frozen_share']:.3f} | {item['carried_share']:.3f} | {item['cancellation_ratio']:.3f} | {item['random_top3_share']:.3f} | {rec['median_source_agreement']:.3f} ({rec['channel_count']}) |")
    lines += ["","| Layer | Cross-corpus dominant-source agreement | Frozen heads | Top weighted source→channel edges |","|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        edges=", ".join(f"{row['source']}→{row['channel']}" for row in results[str(layer)]["top_edges"][:6]); lines.append(f"| {layer} | {results[str(layer)]['cross_corpus_source_agreement']:.3f} | {results[str(layer)]['frozen_heads']} | {edges} |")
    lines += ["",f"Maximum integrated factor recomposition error: **{maximum_integral_error:.2e}**. Maximum discrepancy from native float32 attention accumulation/end state: **{maximum_native_error:.2e}**.","","## Concrete routes",""]
    for layer in config["layers"]:
        lines.append(f"### Layer {layer}")
        for example in results[str(layer)]["concrete"]:
            detail="; ".join(f"N{row['channel']} {row['operand_side']}←{row['dominant_source']} ({row['source_share']:.2f})" for row in example["channels"]); lines.append(f"- **{example['tokens']}**: {detail}. Context: `{example['context']}`")
    lines += ["","Absolute source shares measure concentration but can hide cancellation; the cancellation ratio is reported alongside them. Head identities are native architectural coordinates, while the straight-path allocation is one principled finite attribution rather than a claim of path uniqueness."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
