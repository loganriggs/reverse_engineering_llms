#!/usr/bin/env python3
"""Symmetry-aware factor roles of dynamically selected Layer-0/1 neurons."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_final_bilinear_readout import bilinear_terms
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_dynamic_neuron_roles.json")
OUTPUT=TENSOR_ROOT/"layer01_dynamic_neuron_roles.json"
REPORT=TENSOR_ROOT/"LAYER01_DYNAMIC_NEURON_ROLES.md"
FIGURE=TENSOR_ROOT/"layer01_dynamic_neuron_roles.png"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def role_statistics(left,right,quadratic,epsilon=1e-30):
    left_abs,right_abs,quadratic_abs=left.abs(),right.abs(),quadratic.abs(); cross=left_abs+right_abs
    dominance=torch.maximum(left_abs,right_abs)/cross.clamp_min(epsilon)
    side=left_abs>=right_abs
    quadratic_share=quadratic_abs/(cross+quadratic_abs).clamp_min(epsilon)
    cancellation=(left*right)<0
    return dominance,side,quadratic_share,cancellation


def pair_aggregate(rows,rank):
    scores=collections.defaultdict(float); left=collections.defaultdict(float); right=collections.defaultdict(float); signs=collections.defaultdict(float)
    for row in rows:
        for index,score,lvalue,rvalue,sign in zip(row["indices"],row["scores"],row["left_abs"],row["right_abs"],row["signs"]):
            scores[index]+=score; left[index]+=lvalue; right[index]+=rvalue; signs[index]+=sign
    selected=sorted(scores,key=scores.get,reverse=True)[:rank]
    return {index:{"side":left[index]>=right[index],"sign":signs[index]>=0} for index in selected}


def token_tails(embedding,factor,tokenizer,count,orientation):
    score=embedding@(orientation*factor); positive=score.topk(count).indices.tolist(); negative=(-score).topk(count).indices.tolist()
    decode=lambda ids:[{"id":index,"token":tokenizer.decode([index]),"score":float(score[index])} for index in ids]
    return {"positive":decode(positive),"negative":decode(negative)}


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); bilinear=json.loads(Path(config["source_config"]).read_text()); final_config=json.loads(Path(bilinear["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source); new_config=json.loads(Path(config["new_pair_config"]).read_text()); edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}; tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True); embedding=F.rms_norm(model.embedding.weight.float(),(model.config.n_embd,)); results={}
    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); records=[]; concrete=[]; random_indices=torch.randperm(decoder.shape[1],generator=torch.Generator(device="cpu").manual_seed(config["seed"]+100+layer))[:config["rank"]].cuda()
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    pair=f"{source_token}->{target_token}"
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token; base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]; base_state=F.rms_norm(base["pre"][0,final_position].float(),(model.config.n_embd,)); counter_state=F.rms_norm(counter["pre"][0,final_position].float(),(model.config.n_embd,)); terms=bilinear_terms(block,base_state,counter_state); full=sum(terms.values()); target_write=F.linear(full,decoder); denominator=target_write.square().sum().clamp_min(1e-30); contribution=full*(target_write@decoder)/denominator; indices=contribution.abs().topk(config["rank"]).indices; dominance,side,quadratic_share,cancellation=role_statistics(terms["left_cross"],terms["right_cross"],terms["quadratic"])
                        def baseline(use_indices):
                            return {"single_sided":float((dominance[use_indices]>=config["dominance_threshold"]).float().mean()),"balanced":float((dominance[use_indices]<=config["balanced_threshold"]).float().mean()),"quadratic_share":float(quadratic_share[use_indices].median()),"cancellation":float(cancellation[use_indices].float().mean())}
                        activation_indices=full.abs().topk(config["rank"]).indices; all_indices=torch.arange(len(full),device=full.device)
                        record={"corpus":corpus,"family":family_name,"pair":pair,"context":context_index,"indices":indices.cpu().tolist(),"scores":contribution[indices].abs().cpu().tolist(),"left_abs":terms["left_cross"][indices].abs().cpu().tolist(),"right_abs":terms["right_cross"][indices].abs().cpu().tolist(),"dominance":dominance[indices].cpu().tolist(),"sides":side[indices].cpu().tolist(),"quadratic_share":quadratic_share[indices].cpu().tolist(),"cancellation":cancellation[indices].cpu().tolist(),"signs":torch.sign(contribution[indices]).cpu().tolist(),"positive_fraction":float((contribution[indices]>0).float().mean()),"selection_baselines":{"contribution":baseline(indices),"activation":baseline(activation_indices),"random":baseline(random_indices),"all":baseline(all_indices)}}; records.append(record)
                        if corpus=="fineweb" and context_index==0 and pair in config["concrete_pairs"]:
                            rows_out=[]
                            for channel in indices[:config["concrete_channels"]].tolist():
                                left_dominant=bool(side[channel]); edit_factor=block.mlp.Left.weight.float()[channel] if left_dominant else block.mlp.Right.weight.float()[channel]; gate_factor=block.mlp.Right.weight.float()[channel] if left_dominant else block.mlp.Left.weight.float()[channel]; edit_delta=(block.mlp.Left(counter_state)[channel]-block.mlp.Left(base_state)[channel]) if left_dominant else (block.mlp.Right(counter_state)[channel]-block.mlp.Right(base_state)[channel]); orientation=1. if float(edit_delta)>=0 else -1.
                                rows_out.append({"channel":channel,"local_projection":float(contribution[channel]),"product_delta":float(full[channel]),"left_cross":float(terms["left_cross"][channel]),"right_cross":float(terms["right_cross"][channel]),"quadratic":float(terms["quadratic"][channel]),"dominance":float(dominance[channel]),"edit_side":"left" if left_dominant else "right","edit_factor_token_tails":token_tails(embedding,edit_factor,tokenizer,config["token_tails"],orientation),"gate_factor_token_tails":token_tails(embedding,gate_factor,tokenizer,config["token_tails"],orientation)})
                            concrete.append({"pair":pair,"tokens":f"{tokenizer.decode([source_token])}->{tokenizer.decode([target_token])}","context":tokenizer.decode(raw[max(0,edit_position-12):final_position+1]),"channels":rows_out})
                    print("dynamic roles",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={}; recurring={}
        for corpus in fixed["evaluation"]:
            selected=[row for row in records if row["corpus"]==corpus]; summaries[corpus]={"single_sided_fraction":float(np.mean([value>=config["dominance_threshold"] for row in selected for value in row["dominance"]])),"balanced_fraction":float(np.mean([value<=config["balanced_threshold"] for row in selected for value in row["dominance"]])),"quadratic_share":median([value for row in selected for value in row["quadratic_share"]]),"cancellation_fraction":float(np.mean([value for row in selected for value in row["cancellation"]])),"positive_fraction":float(np.mean([row["positive_fraction"] for row in selected])),"selection_baselines":{selection:{metric:median([row["selection_baselines"][selection][metric] for row in selected]) for metric in ("single_sided","balanced","quadratic_share","cancellation")} for selection in ("contribution","activation","random","all")}}
            by_channel=collections.defaultdict(list)
            for row in selected:
                for index,side_value,dominance_value,sign in zip(row["indices"],row["sides"],row["dominance"],row["signs"]): by_channel[index].append((side_value,dominance_value,sign))
            stable=[]
            for channel,values in by_channel.items():
                if len(values)>=config["recurrence_minimum"]:
                    sides=[value[0] for value in values]; stable.append(max(np.mean(sides),1-np.mean(sides)))
            recurring[corpus]={"channel_count":len(stable),"median_side_agreement":median(stable) if stable else None}
        pair_rows=[]
        pairs=sorted({row["pair"] for row in records})
        for pair in pairs:
            left=pair_aggregate([row for row in records if row["corpus"]=="fineweb" and row["pair"]==pair],config["rank"]); right=pair_aggregate([row for row in records if row["corpus"]=="tinystories" and row["pair"]==pair],config["rank"]); intersection=set(left)&set(right); union=set(left)|set(right); pair_rows.append({"pair":pair,"jaccard":len(intersection)/len(union),"intersection":len(intersection),"side_agreement":float(np.mean([left[index]["side"]==right[index]["side"] for index in intersection])) if intersection else None,"sign_agreement":float(np.mean([left[index]["sign"]==right[index]["sign"] for index in intersection])) if intersection else None})
        cross={key:median([row[key] for row in pair_rows if row[key] is not None]) for key in ("jaccard","intersection","side_agreement","sign_agreement")}; results[str(layer)]={"summaries":summaries,"recurring":recurring,"cross_corpus":cross,"pair_rows":pair_rows,"concrete":concrete,"records":records}
    def summary(layer,corpus): return results[str(layer)]["summaries"][corpus]
    hypotheses={
        "R0_top_channels_are_factor_asymmetric":{"supported":all(summary(l,c)["single_sided_fraction"]>=.5 for l in config["layers"] for c in fixed["evaluation"])},
        "R1_balanced_channels_are_a_minority":{"supported":all(summary(l,c)["balanced_fraction"]<=.3 for l in config["layers"] for c in fixed["evaluation"])},
        "R2_recurring_channels_keep_their_factor_role":{"supported":all(results[str(l)]["recurring"][c]["median_side_agreement"] is not None and results[str(l)]["recurring"][c]["median_side_agreement"]>=.75 for l in config["layers"] for c in fixed["evaluation"])},
        "R3_channel_identity_is_contextual":{"supported":all(results[str(l)]["cross_corpus"]["jaccard"]<=.25 for l in config["layers"])},
        "R4_factor_role_transfers_when_identity_recurs":{"supported":all(results[str(l)]["cross_corpus"]["side_agreement"]>=.75 for l in config["layers"])},
        "R5_contribution_sign_transfers_when_identity_recurs":{"supported":all(results[str(l)]["cross_corpus"]["sign_agreement"]>=.65 for l in config["layers"])},
        "R6_quadratic_term_is_negligible_per_channel":{"supported":all(summary(l,c)["quadratic_share"]<.05 for l in config["layers"] for c in fixed["evaluation"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2)); lines=["# Factor roles of dynamically selected Layer-0/1 neurons","","For the top-256 native channels by exact local-write contribution, each finite product change is split into scale-invariant left-cross, right-cross, and quadratic terms. Dominance is invariant to reciprocal factor scaling; swapping a neuron's factors flips the side name but preserves dominance and cross-example agreement. The same lexical pairs are compared across FineWeb and TinyStories.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Single-sided ≥.8 | Balanced ≤.6 | Quadratic share | Cancellation | Recurring channels / side agreement |","|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=summary(layer,corpus); rec=results[str(layer)]["recurring"][corpus]; lines.append(f"| {layer} | {corpus} | {item['single_sided_fraction']:.3f} | {item['balanced_fraction']:.3f} | {item['quadratic_share']:.4f} | {item['cancellation_fraction']:.3f} | {rec['channel_count']} / {rec['median_side_agreement']:.3f} |")
    lines += ["","| Layer | Cross-corpus pair Jaccard | Intersection size | Side agreement | Sign agreement |","|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        item=results[str(layer)]["cross_corpus"]; lines.append(f"| {layer} | {item['jaccard']:.3f} | {item['intersection']:.0f} | {item['side_agreement']:.3f} | {item['sign_agreement']:.3f} |")
    lines += ["","## Post-run selection-artifact red-team","","The locked role hypotheses selected channels by exact local contribution. The table compares that population with activation-selected, fixed-random, and all channels. This diagnostic was added after the first run.","","| Layer | Corpus | Single-sided contribution / activation / random / all | Quadratic share contribution / all | Positive contribution fraction |","|---:|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=summary(layer,corpus); b=item["selection_baselines"]; lines.append(f"| {layer} | {corpus} | {b['contribution']['single_sided']:.3f} / {b['activation']['single_sided']:.3f} / {b['random']['single_sided']:.3f} / {b['all']['single_sided']:.3f} | {b['contribution']['quadratic_share']:.4f} / {b['all']['quadratic_share']:.4f} | {item['positive_fraction']:.3f} |")
    lines += ["","## Concrete FineWeb examples",""]
    for layer in config["layers"]:
        lines.append(f"### Layer {layer}")
        for example in results[str(layer)]["concrete"]:
            lines += ["",f"- **{example['tokens']}** in `{example['context']}`"]
            for row in example["channels"]:
                edit_pos=", ".join(repr(item["token"]) for item in row["edit_factor_token_tails"]["positive"][:3]); gate_pos=", ".join(repr(item["token"]) for item in row["gate_factor_token_tails"]["positive"][:3]); lines.append(f"  - channel {row['channel']}: projection {row['local_projection']:+.3f}, {row['edit_side']} dominance {row['dominance']:.3f}, cross terms {row['left_cross']:+.3g}/{row['right_cross']:+.3g}; embedding-nearest edit/gate tails {edit_pos} / {gate_pos}")
    lines += ["","Embedding-tail labels are weight-derived clues, not validated semantic names: the MLP reads contextual RMS-normalized states, not isolated token embeddings. The causal evidence is the exact cross-term value and downstream channel contribution."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained"); cells=[(l,c) for l in config["layers"] for c in fixed["evaluation"]]; x=np.arange(len(cells)); labels=[f"L{l} {c[:4]}" for l,c in cells]
    axes[0].bar(x-.18,[summary(l,c)["single_sided_fraction"] for l,c in cells],.36,label="single-sided"); axes[0].bar(x+.18,[summary(l,c)["balanced_fraction"] for l,c in cells],.36,label="balanced"); axes[0].set_xticks(x,labels); axes[0].legend(); axes[0].set_ylabel("selected-channel fraction"); axes[0].grid(axis="y",alpha=.25)
    axes[1].bar(np.arange(2)-.18,[results[str(l)]["cross_corpus"]["side_agreement"] for l in config["layers"]],.36,label="factor side"); axes[1].bar(np.arange(2)+.18,[results[str(l)]["cross_corpus"]["sign_agreement"] for l in config["layers"]],.36,label="contribution sign"); axes[1].set_xticks(np.arange(2),[f"Layer {l}" for l in config["layers"]]); axes[1].set_ylim(0,1); axes[1].legend(); axes[1].grid(axis="y",alpha=.25); fig.savefig(FIGURE,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
