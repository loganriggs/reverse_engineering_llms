#!/usr/bin/env python3
"""Screen later native attention edges for consumers of the L8H2 spelling edit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_two_hop_circuit import edge_write
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_boundary_style_downstream_screen.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_downstream_screen.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_DOWNSTREAM_SCREEN.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_downstream_screen.png"


@torch.inference_mode()
def upstream_states(model, original, swapped, cue, boundary):
    dimension=model.config.n_embd; heads=model.config.n_head; batch=len(original); paired=torch.arange(batch,device="cuda")
    ids=torch.as_tensor(np.concatenate([original,swapped]),device="cuda"); x0=F.rms_norm(model.embedding(ids),(dimension,)); x=x0; bus=None
    for layer,block in enumerate(model.blocks[:9]):
        mixed=block.lambdas[0]*x+block.lambdas[1]*x0; details=attention_details(block.attn,F.rms_norm(mixed,(dimension,)),bus,model.config,model.tt)
        pre=mixed+details["write"]; state=F.rms_norm(pre,(dimension,)); mlp=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
        if layer<8: x=pre+mlp; bus=details["value_bus"]; continue
        edge0=edge_write(details,block,2,boundary,cue,paired,dimension,heads); edge1=edge_write(details,block,2,boundary,cue,paired+batch,dimension,heads)
        native=details["write"][:batch].float(); patched=native.clone(); patched[paired,boundary]+=edge1-edge0
        attention=torch.cat([native,patched]); mixed=torch.cat([mixed[:batch].float(),mixed[:batch].float()]); x0=torch.cat([x0[:batch].float(),x0[:batch].float()]); bus=torch.cat([details["value_bus"][:batch].float(),details["value_bus"][:batch].float()])
        state=F.rms_norm(mixed+attention,(dimension,)); x=mixed+attention+block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
        return x,x0,bus


def accumulator(): return {"native":[],"frozen":[],"mediated":[],"random_mediated":[]}


@torch.inference_mode()
def collect(model,raw,records,contrasts,config):
    dimension=model.config.n_embd; heads=model.config.n_head; destination=127
    names=[f"h{h}" for h in range(heads)]+["all_boundary","full_destination"]
    stats={(layer,name):accumulator() for layer in config["layers"] for name in names}; consistency=0.0; reference_native=None
    for start in range(0,len(records),config["batch_size"]):
        batch=records[start:start+config["batch_size"]]; size=len(batch); paired=torch.arange(size,device="cuda")
        original=np.stack([raw[r[0],:128].copy() for r in batch]); swapped=original.copy()
        for i,r in enumerate(batch): swapped[i,r[3]]=r[5]
        cue=torch.as_tensor([r[3] for r in batch],device="cuda"); boundary=torch.as_tensor([r[1] for r in batch],device="cuda")
        x,x0,bus=upstream_states(model,original,swapped,cue,boundary); direction=torch.as_tensor([1 if not r[4] else -1 for r in batch],device="cuda",dtype=torch.float32)
        batch_reference=None
        for layer in config["layers"]:
            block=model.blocks[layer]; mixed=block.lambdas[0]*x+block.lambdas[1]*x0; details=attention_details(block.attn,F.rms_norm(mixed,(dimension,)),bus,model.config,model.tt)
            rows=torch.arange(2*size,device="cuda"); boundary2=torch.cat([boundary,boundary]); dest2=torch.full_like(boundary2,destination)
            edges=[edge_write(details,block,h,dest2,boundary2,rows,dimension,heads).reshape(2,size,dimension) for h in range(heads)]
            variants=[details["write"][:size].float(),details["write"][size:].float()]
            for h in range(heads):
                modified=details["write"][size:].float().clone(); modified[paired,destination]+=edges[h][0]-edges[h][1]; variants.append(modified)
            modified=details["write"][size:].float().clone(); modified[paired,destination]+=sum(edge[0]-edge[1] for edge in edges); variants.append(modified)
            modified=details["write"][size:].float().clone(); modified[paired,destination]=details["write"][:size, destination].float(); variants.append(modified)
            attention=torch.cat(variants); repeats=len(variants); mixedv=torch.cat([mixed[:size],mixed[size:]]+[mixed[size:]]*(repeats-2)); x0v=torch.cat([x0[:size],x0[size:]]+[x0[size:]]*(repeats-2)); busv=torch.cat([details["value_bus"][:size],details["value_bus"][size:]]+[details["value_bus"][size:]]*(repeats-2)).float()
            state=F.rms_norm(mixedv+attention,(dimension,)); suffix_x=mixedv+attention+block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias; suffix_bus=busv
            for suffix in model.blocks[layer+1:]: suffix_x,suffix_bus=suffix(suffix_x,suffix_bus,x0v)
            logits=30*torch.tanh(model.lm_head(F.rms_norm(suffix_x,(dimension,))[:,destination]).float()/30); logits=logits.reshape(repeats,size,-1)
            if batch_reference is None: batch_reference=logits[:2].clone()
            else: consistency=max(consistency,float((logits[:2]-batch_reference).abs().max()))
            scores=logits@contrasts; native=direction[:,None]*(scores[1]-scores[0])
            for index,name in enumerate(names,start=2):
                frozen=direction[:,None]*(scores[index]-scores[0]); mediated=native-frozen; row=stats[(layer,name)]
                row["native"].append(native[:,0].cpu()); row["frozen"].append(frozen[:,0].cpu()); row["mediated"].append(mediated[:,0].cpu()); row["random_mediated"].append(mediated[:,1:].cpu())
            pre=mixed+details["write"]; native_state=F.rms_norm(pre,(dimension,)); x=pre+block.mlp.Down(block.mlp.Left(native_state)*block.mlp.Right(native_state))+block.mlp.Down_bias; bus=details["value_bus"]
    result={}
    for (layer,name),row in stats.items():
        native=torch.cat(row["native"]); frozen=torch.cat(row["frozen"]); mediated=torch.cat(row["mediated"]); random=torch.cat(row["random_mediated"]).mean(0); native_mean=float(native.mean()); mean=float(mediated.mean())
        result[f"L{layer}_{name}"]={"layer":layer,"program":name,"examples":len(native),"native_mean":native_mean,"frozen_mean":float(frozen.mean()),"mediated_mean":mean,"mediation_fraction":mean/max(abs(native_mean),1e-30),"mediated_win":float((mediated>0).float().mean()),"random_percentile":float((random<mean).float().mean())}
    return result,consistency


def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary_flags=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool)
    pairs=[(tokenizer.encode(a,add_special_tokens=False)[0],tokenizer.encode(b,add_special_tokens=False)[0]) for a,b in base["spelling_pairs"]]; targets=set(range(config["fold"]*config["target_pairs_per_fold"],(config["fold"]+1)*config["target_pairs_per_fold"])); target_pairs=[p for i,p in enumerate(pairs) if i in targets]; cues=[p for i,p in enumerate(pairs) if i not in targets]
    begin,end=base["splits"][config["split"]]; raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary_flags,cues,counts,127,config["examples_per_label"],config["seed"])
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    contrasts=make_contrasts(model.lm_head.weight.shape[0],target_pairs,config["random_contrasts"],config["seed"],"cuda")
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("screen",len(records),flush=True); results,consistency=collect(model,raw,records,contrasts,config)
    individuals=[row for row in results.values() if row["program"].startswith("h")]; ranked=sorted(individuals,key=lambda r:r["mediated_mean"],reverse=True); rank_l9h8=next(i+1 for i,r in enumerate(ranked) if r["layer"]==9 and r["program"]=="h8")
    s0=consistency<=1e-4; s1=rank_l9h8<=5; s2=any(r["layer"]>9 and r["mediation_fraction"]>=.10 for r in individuals)
    s3=False; s4=False
    for layer in config["layers"]:
        best=max(results[f"L{layer}_h{h}"]["mediation_fraction"] for h in range(9)); allb=results[f"L{layer}_all_boundary"]["mediation_fraction"]; full=results[f"L{layer}_full_destination"]["mediation_fraction"]
        s3|=allb>=best+.20; s4|=full>=allb+.20
    hypotheses={"S0_native_endpoints_are_consistent":{"supported":s0,"max_logit_error":consistency},"S1_known_L9H8_is_rediscovered":{"supported":s1,"rank":rank_l9h8},"S2_additional_later_consumer_exists":{"supported":s2},"S3_boundary_edges_explain_each_attention_stage":{"supported":s3},"S4_destination_attention_has_nonboundary_residual":{"supported":s4}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"ranked_individuals":ranked,"results":results},indent=2,allow_nan=False))
    lines=["# Screen of downstream consumers of the L8H2 spelling edit","","After the finite L8H2 cue-edge replacement, every boundary-to-destination edge in Layers 9--17 is individually frozen to its pre-edit write and propagated through the full suffix. All boundary edges and the complete destination-attention change are additional layer controls. This discovery screen uses one locked spelling fold/split; selected routes require disjoint replication.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Rank | Edge | mediated fraction | mean | win | random percentile |","|---:|---|---:|---:|---:|---:|"]
    for rank,row in enumerate(ranked[:16],1): lines.append(f"| {rank} | L{row['layer']}H{row['program'][1:]} | {row['mediation_fraction']:.3f} | {row['mediated_mean']:+.4g} | {row['mediated_win']:.3f} | {row['random_percentile']:.3f} |")
    lines += ["","| Layer | best head | all boundary | full destination attention |","|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        best=max(results[f"L{layer}_h{h}"]["mediation_fraction"] for h in range(9)); lines.append(f"| {layer} | {best:.3f} | {results[f'L{layer}_all_boundary']['mediation_fraction']:.3f} | {results[f'L{layer}_full_destination']['mediation_fraction']:.3f} |")
    lines += ["","## Interpretation boundary","","Individual freeze effects are non-additive and can be negative. This screen ranks candidate consumers; it does not assign percentages that sum to one. A later edge can redundantly read a state already consumed earlier, and freezing an earlier edge changes the states presented to later routes. The next gate is a prospectively frozen multi-edge program on disjoint folds and splits."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(12,4.5),layout="constrained"); xs=[];ys=[];labels=[]
    for layer in config["layers"]:
        for h in range(9): xs.append(layer+(h-4)*.07); ys.append(results[f"L{layer}_h{h}"]["mediation_fraction"]); labels.append(h)
    scatter=ax.scatter(xs,ys,c=labels,cmap="tab10",s=28); ax.axhline(0,color="black",linewidth=.8); ax.set(xlabel="Layer (head jitter)",ylabel="Individual mediated fraction",title="Later boundary-edge consumers of the spelling edit"); ax.grid(alpha=.2); fig.colorbar(scatter,ax=ax,label="head"); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"top":ranked[:12]},indent=2)); print(REPORT)


if __name__=="__main__": main()
