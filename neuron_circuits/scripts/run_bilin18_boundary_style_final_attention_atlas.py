#!/usr/bin/env python3
"""Allocate final spelling effects across Layer-17 heads and source relations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts,select_records
from run_bilin18_boundary_style_downstream_screen import upstream_states
from run_bilin18_layer01_supervised_feature_census import FEATURES,vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details,load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT

CONFIG=Path("configs/bilin18_boundary_style_final_attention_atlas.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_final_attention_atlas.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_FINAL_ATTENTION_ATLAS.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_final_attention_atlas.png"
RELATIONS=("before_cue","cue","between","boundary","after_boundary","destination")

def relation_masks(length,cue,boundary,destination):
    p=torch.arange(length,device=cue.device)[None]
    return {"before_cue":p<cue[:,None],"cue":p==cue[:,None],"between":(p>cue[:,None])&(p<boundary[:,None]),"boundary":p==boundary[:,None],"after_boundary":(p>boundary[:,None])&(p<destination),"destination":(p==destination)&(boundary[:,None]!=destination)}

def score_and_gradient(state,weights,token_weights):
    eps=torch.finfo(torch.float32).eps; radius=(state.square().mean(-1,keepdim=True)+eps).sqrt(); normalized=state/radius; raw=normalized@weights.T; capped=30*torch.tanh(raw/30); score=capped@token_weights
    derivative=token_weights[None]/torch.cosh(raw/30).square(); grad_normalized=derivative@weights; grad=grad_normalized/radius-state*(grad_normalized*state).mean(-1,keepdim=True)/radius.pow(3)
    return score,grad

@torch.inference_mode()
def collect(model,raw,records,contrast,config):
    dimension=model.config.n_embd; heads=model.config.n_head; head_dim=dimension//heads; destination=127; token_index=torch.nonzero(contrast).flatten(); token_weights=contrast[token_index].double(); weights=model.lm_head.weight[token_index].double(); nodes,quadrature=np.polynomial.legendre.leggauss(config["quadrature_points"]); nodes=(nodes+1)/2; quadrature=quadrature/2
    names=["carried","mlp"]+[f"h{h}_{r}" for h in range(heads) for r in RELATIONS]+["attention_numeric_residual"]; sums={name:0.0 for name in names}; count=0; max_error=0.0
    for start in range(0,len(records),config["batch_size"]):
        batch=records[start:start+config["batch_size"]]; size=len(batch); original=np.stack([raw[r[0],:128].copy() for r in batch]); swapped=original.copy()
        for i,r in enumerate(batch): swapped[i,r[3]]=r[5]
        cue=torch.as_tensor([r[3] for r in batch],device="cuda"); boundary=torch.as_tensor([r[1] for r in batch],device="cuda"); direction=torch.as_tensor([1 if not r[4] else -1 for r in batch],device="cuda",dtype=torch.float64)
        x,x0,bus=upstream_states(model,original,swapped,cue,boundary)
        for layer in range(9,18):
            block=model.blocks[layer]; mixed=block.lambdas[0]*x+block.lambdas[1]*x0; details=attention_details(block.attn,F.rms_norm(mixed,(dimension,)),bus,model.config,model.tt); pre=mixed+details["write"]; state=F.rms_norm(pre,(dimension,)); mlp=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
            if layer<17: x=pre+mlp; bus=details["value_bus"]; continue
            pattern=(details["score1"]*details["score2"])[:,:,destination,:].double(); values=details["mixed_value"].double().permute(0,2,1,3); projection=block.attn.c_proj.weight.double().T.reshape(heads,head_dim,dimension); edge=torch.einsum("bhs,bhsd,hdk->bhsk",pattern,values,projection).reshape(2,size,heads,original.shape[1],dimension); edge_delta=edge[1]-edge[0]; masks=relation_masks(original.shape[1],cue,boundary,destination)
            components={"carried":mixed[size:,destination].double()-mixed[:size,destination].double(),"mlp":mlp[size:,destination].double()-mlp[:size,destination].double()}
            for h in range(heads):
                for relation,mask in masks.items(): components[f"h{h}_{relation}"]=(edge_delta[:,h]*mask[:,:,None]).sum(1)
            attention_delta=details["write"][size:,destination].double()-details["write"][:size,destination].double(); components["attention_numeric_residual"]=attention_delta-sum(value for name,value in components.items() if name.startswith("h"))
            base=mixed[:size,destination].double()+details["write"][:size,destination].double()+mlp[:size,destination].double(); target=mixed[size:,destination].double()+details["write"][size:,destination].double()+mlp[size:,destination].double(); delta=target-base
            contributions={name:torch.zeros(size,device="cuda",dtype=torch.float64) for name in names}
            for node,weight in zip(nodes,quadrature):
                _,grad=score_and_gradient(base+float(node)*delta,weights,token_weights)
                for name,value in components.items(): contributions[name]+=float(weight)*(grad*value).sum(-1)
            score0,_=score_and_gradient(base,weights,token_weights); score1,_=score_and_gradient(target,weights,token_weights); target_score=score1-score0; reconstructed=sum(contributions.values()); max_error=max(max_error,float((reconstructed-target_score).norm()/target_score.norm().clamp_min(1e-30)))
            for name,value in contributions.items(): sums[name]+=float((direction*value).sum())
            count+=size; break
    means={name:value/count for name,value in sums.items()}; total=sum(means.values()); attention={name:value for name,value in means.items() if name.startswith("h")}; ranked=sorted(attention.items(),key=lambda x:x[1],reverse=True)
    boundary_names={f"h{h}_boundary" for h in range(heads)}
    nonboundary={name:value for name,value in attention.items() if name not in boundary_names}; positive=sum(max(v,0) for v in nonboundary.values()); top=sorted((max(v,0),name) for name,v in nonboundary.items())[::-1]
    relation_means={relation:sum(attention[f"h{h}_{relation}"] for h in range(heads)) for relation in RELATIONS}
    return {"examples":count,"recomposition_error":max_error,"total_mean":total,"component_means":means,"ranked_attention":[{"component":name,"mean":value} for name,value in ranked],"top_nonboundary":top[0][1],"top_nonboundary_share":top[0][0]/max(positive,1e-30),"top4_nonboundary_share":sum(v for v,_ in top[:4])/max(positive,1e-30),"relation_means":relation_means,"head_means":{str(h):sum(value for name,value in attention.items() if name.startswith(f"h{h}_")) for h in range(heads)}}

def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary_flags=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool); pairs=[(tokenizer.encode(a,add_special_tokens=False)[0],tokenizer.encode(b,add_special_tokens=False)[0]) for a,b in base["spelling_pairs"]]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results=[]
    for fold in range(config["folds"]):
        targets=set(range(fold*config["target_pairs_per_fold"],(fold+1)*config["target_pairs_per_fold"])); target_pairs=[p for i,p in enumerate(pairs) if i in targets]; cues=[p for i,p in enumerate(pairs) if i not in targets]; contrast=make_contrasts(model.lm_head.weight.shape[0],target_pairs,0,config["seed"]+fold,"cuda")[:,0]
        for split,(begin,end) in enumerate(base["splits"]):
            raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary_flags,cues,counts,127,config["examples_per_label_per_split"],config["seed"]+fold*10+split); print("attention atlas",fold,split,len(records),flush=True); results.append({"fold":fold,"split":split,**collect(model,raw,records,contrast,config)})
    a0=max(r["recomposition_error"] for r in results)<=1e-5; ranks=[]
    for r in results: ranks.append(next(i+1 for i,row in enumerate(r["ranked_attention"]) if row["component"]=="h2_boundary"))
    a1=all(rank<=5 for rank in ranks); a2=all(r["top_nonboundary_share"]<=.5 for r in results); a3=all(r["top4_nonboundary_share"]<.9 for r in results); stable=sum(results[2*f]["top_nonboundary"].split("_",1)[1]==results[2*f+1]["top_nonboundary"].split("_",1)[1] for f in range(config["folds"])); a4=stable>=max(1,(2*config["folds"]+2)//3)
    hypotheses={"A0_integrated_component_atlas_recomposes":{"supported":a0,"max_error":max(r["recomposition_error"] for r in results)},"A1_known_L17H2_boundary_is_recovered":{"supported":a1,"ranks":ranks},"A2_nonboundary_attention_is_not_one_component":{"supported":a2},"A3_nonboundary_attention_is_broad":{"supported":a3},"A4_top_nonboundary_relation_replicates":{"supported":a4,"folds":stable},"A5_distributed_final_attention_ontology_is_supported":{"supported":all((a0,a1,a2,a3,a4))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Layer-17 head/source atlas for the spelling edit","","Every changed Layer-17 attention edge is grouped by native head and its source relation to the spelling cue and boundary. A 32-point exact output-gradient line integral allocates the finite spelling contrast over those edge groups, carried residual, final MLP, and the measured float32 attention residual.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Fold/split | H2-boundary rank | top nonboundary component/share | top-4 share | relation means |","|---|---:|---:|---:|---:|"]
    for r,rank in zip(results,ranks): lines.append(f"| {r['fold']}/{r['split']} | {rank} | {r['top_nonboundary']} / {r['top_nonboundary_share']:.3f} | {r['top4_nonboundary_share']:.3f} | "+", ".join(f"{k}:{v:+.3g}" for k,v in r["relation_means"].items())+" |")
    lines += ["","## Interpretation boundary","","The line-integral allocation is exact for the declared spelling contrast and native finite state segment. It is an effect decomposition, not a claim that each head/relation group independently stores a feature. A broad positive nonboundary support indicates many small terminal reads or query-induced edge changes; a sparse result identifies a shorter typed consumer candidate for finite freezing."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained"); x=np.arange(len(results)); labels=[f"f{r['fold']}s{r['split']}" for r in results]
    for relation in RELATIONS: ax.plot(x,[r["relation_means"][relation] for r in results],marker="o",label=relation)
    ax.set(xticks=x,xticklabels=labels,ylabel="Directed integrated contribution",title="Layer-17 attention source relations"); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"summary":[{"fold":r["fold"],"split":r["split"],"top":r["top_nonboundary"],"top1":r["top_nonboundary_share"],"top4":r["top4_nonboundary_share"]} for r in results]},indent=2)); print(REPORT)

if __name__=="__main__": main()
