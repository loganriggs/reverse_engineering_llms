#!/usr/bin/env python3
"""Factor the terminal use of the L8H2 spelling edit into final residual sources."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_downstream_screen import upstream_states
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_boundary_style_final_source_factorial.json")
OUTPUT=TENSOR_ROOT/"bilin18_boundary_style_final_source_factorial.json"
REPORT=TENSOR_ROOT/"BILIN18_BOUNDARY_STYLE_FINAL_SOURCE_FACTORIAL.md"
FIGURE=TENSOR_ROOT/"bilin18_boundary_style_final_source_factorial.png"
FACTORS=("carried","boundary_attention","nonboundary_attention","mlp")
WORLDS=tuple(itertools.product((0,1),repeat=4))


def shapley(outputs):
    result={}; n=len(FACTORS)
    for index,name in enumerate(FACTORS):
        value=torch.zeros_like(next(iter(outputs.values()))); others=[i for i in range(n) if i!=index]
        for size in range(n):
            weight=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            for subset in itertools.combinations(others,size):
                absent=[0]*n
                for item in subset: absent[item]=1
                present=absent.copy(); present[index]=1
                value=value+weight*(outputs[tuple(present)]-outputs[tuple(absent)])
        result[name]=value
    return result


@torch.inference_mode()
def collect(model,raw,records,contrasts,config):
    dimension=model.config.n_embd; heads=model.config.n_head; head_dim=dimension//heads; destination=127
    accumulation_dtype=torch.float64 if config.get("high_precision",False) else torch.float32
    targets=[]; random_targets=[]; parts={name:[] for name in FACTORS}; random_parts={name:[] for name in FACTORS}; direct={name:[] for name in FACTORS}
    state_error=edge_error=output_error=0.0
    for start in range(0,len(records),config["batch_size"]):
        batch=records[start:start+config["batch_size"]]; size=len(batch); paired=torch.arange(size,device="cuda")
        original=np.stack([raw[r[0],:128].copy() for r in batch]); swapped=original.copy()
        for i,r in enumerate(batch): swapped[i,r[3]]=r[5]
        cue=torch.as_tensor([r[3] for r in batch],device="cuda"); boundary=torch.as_tensor([r[1] for r in batch],device="cuda")
        x,x0,bus=upstream_states(model,original,swapped,cue,boundary)
        for layer in range(9,18):
            block=model.blocks[layer]; mixed=block.lambdas[0]*x+block.lambdas[1]*x0; details=attention_details(block.attn,F.rms_norm(mixed,(dimension,)),bus,model.config,model.tt); pre=mixed+details["write"]; state=F.rms_norm(pre,(dimension,)); mlp=block.mlp.Down(block.mlp.Left(state)*block.mlp.Right(state))+block.mlp.Down_bias
            if layer<17: x=pre+mlp; bus=details["value_bus"]; continue
            pattern=(details["score1"]*details["score2"])[:,:,destination,:].to(accumulation_dtype); values=details["mixed_value"].to(accumulation_dtype).permute(0,2,1,3); projection=block.attn.c_proj.weight.to(accumulation_dtype).T.reshape(heads,head_dim,dimension); edges=torch.einsum("bhs,bhsd,hdk->bhsk",pattern,values,projection).reshape(2,size,heads,original.shape[1],dimension)
            edge_delta=edges[1]-edges[0]; positions=torch.arange(original.shape[1],device="cuda")[None]; mask=positions==boundary[:,None]
            boundary_attention=(edge_delta*mask[:,None,:,None]).sum((1,2)); attention_delta=details["write"][size:,destination].to(accumulation_dtype)-details["write"][:size,destination].to(accumulation_dtype); nonboundary_attention=attention_delta-boundary_attention
            edge_error=max(edge_error,float((edge_delta.sum((1,2))-attention_delta).norm()/attention_delta.norm().clamp_min(1e-30)))
            components={"carried":mixed[size:,destination].to(accumulation_dtype)-mixed[:size,destination].to(accumulation_dtype),"boundary_attention":boundary_attention,"nonboundary_attention":nonboundary_attention,"mlp":mlp[size:,destination].to(accumulation_dtype)-mlp[:size,destination].to(accumulation_dtype)}
            base=mixed[:size,destination].to(accumulation_dtype)+details["write"][:size,destination].to(accumulation_dtype)+mlp[:size,destination].to(accumulation_dtype); target=mixed[size:,destination].to(accumulation_dtype)+details["write"][size:,destination].to(accumulation_dtype)+mlp[size:,destination].to(accumulation_dtype); state_error=max(state_error,float((base+sum(components.values())-target).norm()/ (target-base).norm().clamp_min(1e-30)))
            logits={}
            for world in WORLDS:
                residual=(base+sum(world[i]*components[name] for i,name in enumerate(FACTORS))).float(); logits[world]=30*torch.tanh(model.lm_head(F.rms_norm(residual,(dimension,))).float()/30)@contrasts
            target_score=logits[(1,1,1,1)]-logits[(0,0,0,0)]; decomposition=shapley(logits); output_error=max(output_error,float((sum(decomposition.values())-target_score).norm()/target_score.norm().clamp_min(1e-30)))
            direction=torch.as_tensor([1 if not r[4] else -1 for r in batch],device="cuda",dtype=torch.float32); targets.append((direction*target_score[:,0]).cpu()); random_targets.append((direction[:,None]*target_score[:,1:]).cpu())
            for i,name in enumerate(FACTORS):
                parts[name].append((direction*decomposition[name][:,0]).cpu()); random_parts[name].append((direction[:,None]*decomposition[name][:,1:]).cpu()); singleton=[0]*4; singleton[i]=1; direct[name].append((direction*(logits[tuple(singleton)][:,0]-logits[(0,0,0,0)][:,0])).cpu())
            break
    target=torch.cat(targets); random_target=torch.cat(random_targets).mean(0); total_mean=float(target.mean()); output={}
    for name in FACTORS:
        value=torch.cat(parts[name]); random=torch.cat(random_parts[name]).mean(0); output[name]={"mean":float(value.mean()),"win":float((value>0).float().mean()),"projection_share":float((value.double()*target.double()).sum()/target.double().square().sum().clamp_min(1e-30)),"random_percentile":float((random<value.mean()).float().mean()),"direct_mean":float(torch.cat(direct[name]).mean())}
    output["attention_combined"]={"mean":output["boundary_attention"]["mean"]+output["nonboundary_attention"]["mean"],"projection_share":output["boundary_attention"]["projection_share"]+output["nonboundary_attention"]["projection_share"]}
    return {"examples":len(target),"state_recomposition_error":state_error,"edge_recomposition_error":edge_error,"output_recomposition_error":output_error,"total":{"mean":total_mean,"win":float((target>0).float().mean()),"random_percentile":float((random_target<total_mean).float().mean())},"sources":output}


def main():
    config=json.loads(CONFIG.read_text()); base=json.loads(Path(config["source_config"]).read_text()); tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True); corpus=np.load(base["corpus"],mmap_mode="r"); counts=np.bincount(np.asarray(corpus).reshape(-1),minlength=tokenizer.vocab_size); flags=np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)]); boundary=flags[:,FEATURES.index("newline")].astype(bool)|flags[:,FEATURES.index("period")].astype(bool); pairs=[(tokenizer.encode(a,add_special_tokens=False)[0],tokenizer.encode(b,add_special_tokens=False)[0]) for a,b in base["spelling_pairs"]]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results=[]
    for fold in range(config["folds"]):
        targets=set(range(fold*config["target_pairs_per_fold"],(fold+1)*config["target_pairs_per_fold"])); target_pairs=[p for i,p in enumerate(pairs) if i in targets]; cues=[p for i,p in enumerate(pairs) if i not in targets]; contrasts=make_contrasts(model.lm_head.weight.shape[0],target_pairs,config["random_contrasts"],config["seed"]+fold,"cuda")
        for split,(begin,end) in enumerate(base["splits"]):
            raw=np.asarray(corpus[begin:end,:128],dtype=np.int64); records=select_records(raw,boundary,cues,counts,127,config["examples_per_label_per_split"],config["seed"]+fold*10+split); print("final source",fold,split,len(records),flush=True); results.append({"fold":fold,"split":split,**collect(model,raw,records,contrasts,config)})
    max_error=max(max(r["state_recomposition_error"],r["edge_recomposition_error"],r["output_recomposition_error"]) for r in results); f0=max_error<=1e-5; f1=all(r["sources"]["nonboundary_attention"]["mean"]>r["sources"]["boundary_attention"]["mean"] for r in results); f2=all(r["sources"]["attention_combined"]["projection_share"]>=.20 for r in results); f3=all(r["sources"]["carried"]["projection_share"]>=.30 for r in results); f4=all(r["sources"]["mlp"]["projection_share"]>=.10 for r in results)
    hypotheses={"F0_final_source_factorial_recomposes":{"supported":f0,"max_error":max_error},"F1_nonboundary_attention_exceeds_boundary_attention":{"supported":f1},"F2_final_attention_is_material":{"supported":f2},"F3_carried_residual_is_material":{"supported":f3},"F4_final_MLP_is_material":{"supported":f4},"F5_terminal_use_is_multi_branch":{"supported":all((f1,f2,f3,f4))}}
    for row in hypotheses.values(): row["supported"]=bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results},indent=2,allow_nan=False))
    lines=["# Final-source factorial for the L8H2 spelling edit","","At Layer 17's destination, the exact changed output residual is partitioned into carried mixed-state change, boundary-source attention edges, nonboundary attention edges, and the realized final-MLP write change. All sixteen additive source coalitions pass through final RMSNorm, unembedding, and logit capping; finite Shapley values exactly allocate the declared contrast.","","## Locked results",""]
    for name,row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","| Fold/split | total win/mean | carried share | boundary-attn share | nonboundary-attn share | MLP share |","|---|---:|---:|---:|---:|---:|"]
    for r in results: lines.append(f"| {r['fold']}/{r['split']} | {r['total']['win']:.3f}/{r['total']['mean']:+.4g} | {r['sources']['carried']['projection_share']:.3f} | {r['sources']['boundary_attention']['projection_share']:.3f} | {r['sources']['nonboundary_attention']['projection_share']:.3f} | {r['sources']['mlp']['projection_share']:.3f} |")
    lines += ["","## Interpretation boundary","","These are exact allocations at the final additive residual boundary, not exclusive storage locations. The carried state already contains earlier attention and MLP computation, while the realized final-MLP delta depends on carried and attention inputs. The source factorial answers which terminal branches express the intervention under the declared output contrast; it complements rather than replaces the path-specific edge freezes."]
    REPORT.write_text("\n".join(lines)+"\n")
    x=np.arange(len(results)); labels=[f"f{r['fold']}s{r['split']}" for r in results]; fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained"); bottom=np.zeros(len(results))
    for name in FACTORS:
        values=np.asarray([r["sources"][name]["projection_share"] for r in results]); ax.bar(x,values,bottom=bottom,label=name); bottom+=values
    ax.axhline(1,color="black",linewidth=.8); ax.set(xticks=x,xticklabels=labels,ylabel="Projection share",title="Where the spelling edit enters the final logits"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"results":results},indent=2)); print(REPORT)


if __name__=="__main__": main()
