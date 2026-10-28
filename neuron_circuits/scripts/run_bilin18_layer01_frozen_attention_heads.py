#!/usr/bin/env python3
"""Prospective transfer of frozen attention-head rankings at Layers 0 and 1."""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_final_attention_routes import attention_factorization, selected_head_fidelity, static_head_capacity
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer01_frozen_attention_heads.json")
OUTPUT=TENSOR_ROOT/"layer01_frozen_attention_heads.json"
REPORT=TENSOR_ROOT/"LAYER01_FROZEN_ATTENTION_HEADS.md"
FIGURE=TENSOR_ROOT/"layer01_frozen_attention_heads.png"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def jaccard(left,right):
    a,b=set(left),set(right)
    return len(a&b)/len(a|b)


def top_mean(rows,count): return torch.stack(rows).mean(0).topk(count).indices


@torch.inference_mode()
def main():
    config=json.loads(CONFIG.read_text()); attention_config=json.loads(Path(config["source_config"]).read_text()); final_config=json.loads(Path(attention_config["source_config"]).read_text()); new_validation=json.loads(Path(final_config["source_config"]).read_text()); strict=json.loads(Path(new_validation["source_config"]).read_text()); family=json.loads(Path(strict["source_config"]).read_text()); frontier=json.loads(Path(family["source_config"]).read_text()); red=json.loads(Path(frontier["source_config"]).read_text()); tournament=json.loads(Path(red["source_config"]).read_text()); source=json.loads(Path(tournament["source_config"]).read_text()); fixed,_=recover_configs(source)
    saved=json.loads(Path(config["source_records"]).read_text()); new_config=json.loads(Path(config["new_pair_config"]).read_text()); previous=json.loads(Path(config["source_attention_results"]).read_text())
    edit_position,final_position=fixed["edit_position"],fixed["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); arrays={corpus:np.load(spec["path"],mmap_mode="r") for corpus,spec in fixed["evaluation"].items()}
    discovery=[row for row in saved["semantic_records"] if row["corpus"]==config["discovery_corpus"] and row["context_index"] in config["discovery_context_indices"]]
    results={}
    def trace_saved(record,layer):
        spec=fixed["evaluation"][record["corpus"]]; rows=arrays[record["corpus"]][spec["start"]:spec["start"]+spec["rows"]]
        raw=np.asarray(rows[record["row"],record["source_start"]:record["source_start"]+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=int(record["pair"].split("->")[1])
        return forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer],forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]
    for layer in config["layers"]:
        block=model.blocks[layer]; global_rows=[]; family_rows={name:[] for name in new_config["new_families"]}; context_rows={index:[] for index in config["discovery_context_indices"]}; context_family={index:{name:[] for name in new_config["new_families"]} for index in config["discovery_context_indices"]}
        for record in discovery:
            base,counter=trace_saved(record,layer); _,_,head_writes,target,_,_=attention_factorization(block,base,counter,final_position); score=((head_writes@target)/target.square().sum().clamp_min(1e-30)).abs().cpu()
            global_rows.append(score); family_rows[record["family"]].append(score); context_rows[record["context_index"]].append(score); context_family[record["context_index"]][record["family"]].append(score)
        count=config["top_heads"]
        banks={"global":top_mean(global_rows,count).cuda(),"static":static_head_capacity(block).topk(count).indices.cuda()}
        banks.update({f"family_{name}":top_mean(rows,count).cuda() for name,rows in family_rows.items()})
        context_banks={index:top_mean(rows,count).tolist() for index,rows in context_rows.items()}
        context_family_banks={index:{name:top_mean(rows,count).tolist() for name,rows in groups.items()} for index,groups in context_family.items()}
        context_indices=config["discovery_context_indices"]; global_stability=jaccard(context_banks[context_indices[0]],context_banks[context_indices[1]])
        same=[]; different=[]
        for left_name,left in context_family_banks[context_indices[0]].items():
            for right_name,right in context_family_banks[context_indices[1]].items():
                (same if left_name==right_name else different).append(jaccard(left,right))
        records=[]; subsets=list(itertools.combinations(range(block.attn.n_head),count)); subset_values={corpus:[[] for _ in subsets] for corpus in fixed["evaluation"]}
        for corpus_index,(corpus,spec) in enumerate(fixed["evaluation"].items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family_name,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    for context_index,(row_index,start) in enumerate(select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer)[layer]; counter=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)[layer]
                        _,_,head_writes,target,_,_=attention_factorization(block,base,counter,final_position)
                        projection=((head_writes@target)/target.square().sum().clamp_min(1e-30)).abs(); state=projection.topk(count).indices
                        fidelities={name:selected_head_fidelity(target,head_writes,index) for name,index in {**banks,"state":state}.items()}
                        for subset_index,subset in enumerate(subsets):
                            subset_values[corpus][subset_index].append(selected_head_fidelity(target,head_writes,torch.tensor(subset,device=head_writes.device)))
                        records.append({"corpus":corpus,"family":family_name,"pair":f"{source_token}->{target_token}","context":context_index,"fidelity":{"global":fidelities["global"],"family":fidelities[f"family_{family_name}"],"static":fidelities["static"],"state":fidelities["state"]}})
                    print("frozen heads",layer,corpus,family_name,source_token,target_token,flush=True)
        summaries={corpus:{name:median([row["fidelity"][name] for row in records if row["corpus"]==corpus]) for name in ("global","family","static","state")} for corpus in fixed["evaluation"]}
        subset_redteam={}
        global_tuple=tuple(sorted(banks["global"].cpu().tolist()))
        for corpus in fixed["evaluation"]:
            scores=np.asarray([median(values) for values in subset_values[corpus]])
            global_index=subsets.index(global_tuple); best_index=int(scores.argmax())
            subset_redteam[corpus]={"subset_count":len(subsets),"median_fixed_subset":float(np.median(scores)),"global_subset_score":float(scores[global_index]),"global_percentile":float(np.mean(scores<=scores[global_index])),"best_fixed_score":float(scores[best_index]),"best_fixed_heads":list(subsets[best_index]),"global_to_best_gap":float(scores[best_index]-scores[global_index])}
        results[str(layer)]={"banks":{name:value.cpu().tolist() for name,value in banks.items()},"discovery":{"context_global":context_banks,"global_stability":global_stability,"same_family_overlap":median(same),"different_family_overlap":median(different)},"subset_redteam":subset_redteam,"summaries":summaries,"records":records}
    def row(layer,corpus): return results[str(layer)]["summaries"][corpus]
    hypotheses={
        "H0_layer0_has_a_fixed_transport_module":{"supported":all(row(0,c)["global"]>=.75 for c in fixed["evaluation"])},
        "H1_layer1_has_a_fixed_transport_module":{"supported":all(row(1,c)["global"]>=.60 for c in fixed["evaluation"])},
        "H2_family_specific_heads_improve_global":{"supported":all(row(l,c)["family"]-row(l,c)["global"]>=.05 for l in config["layers"] for c in fixed["evaluation"])},
        "H3_calibrated_heads_beat_static_capacity":{"supported":all(row(l,c)["global"]-row(l,c)["static"]>=.30 for l in config["layers"] for c in fixed["evaluation"])},
        "H4_frozen_heads_transfer_ood":{"supported":all(row(l,"tinystories")["global"]>=.8*row(l,"fineweb")["global"] for l in config["layers"])},
        "H5_discovery_head_sets_are_stable":{"supported":all(results[str(l)]["discovery"]["global_stability"]>=.5 for l in config["layers"])},
        "H6_family_head_sets_are_distinct":{"supported":all(results[str(l)]["discovery"]["same_family_overlap"]-results[str(l)]["discovery"]["different_family_overlap"]>=.15 for l in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2))
    lines=["# Frozen attention-head transfer at Layers 0 and 1","","Top-three heads are frozen from the original semantic substitutions in two FineWeb contexts, then evaluated on seventeen new lexical pairs and both FineWeb and TinyStories. Global and family-calibrated rankings are compared with a state-free tensor-capacity ranking and a per-example target oracle.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Frozen global | Frozen family | Static capacity | Per-state | Global head IDs |","|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); lines.append(f"| {layer} | {corpus} | {item['global']:.3f} | {item['family']:.3f} | {item['static']:.3f} | {item['state']:.3f} | {results[str(layer)]['banks']['global']} |")
    lines += ["","| Layer | Context-bank Jaccard | Same-family overlap | Different-family overlap |","|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        item=results[str(layer)]["discovery"]; lines.append(f"| {layer} | {item['global_stability']:.3f} | {item['same_family_overlap']:.3f} | {item['different_family_overlap']:.3f} |")
    lines += ["","## Post-run exhaustive subset red-team","","All 84 native three-head subsets are evaluated on the same prospective examples. This diagnostic was added after observing the frozen result and is not a locked hypothesis.","","| Layer | Corpus | Median subset | Frozen global (percentile) | Best fixed heads / score | Gap to best |","|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=results[str(layer)]["subset_redteam"][corpus]; lines.append(f"| {layer} | {corpus} | {item['median_fixed_subset']:.3f} | {item['global_subset_score']:.3f} ({item['global_percentile']:.3f}) | {item['best_fixed_heads']} / {item['best_fixed_score']:.3f} | {item['global_to_best_gap']:.3f} |")
    lines += ["","A strong frozen-global result identifies a reusable transport module after minimal calibration; a family advantage would additionally support semantic specialization. Failure of the raw capacity ranking only shows that operator magnitude is not an execution map."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axis=plt.subplots(figsize=(9,4.5),layout="constrained"); cells=[(l,c) for l in config["layers"] for c in fixed["evaluation"]]; x=np.arange(len(cells)); labels=[f"L{l} {c[:4]}" for l,c in cells]
    for offset,name in zip((-.3,-.1,.1,.3),("global","family","static","state")): axis.bar(x+offset,[row(l,c)[name] for l,c in cells],.2,label=name)
    axis.set_xticks(x,labels); axis.set_ylabel("top-three attention fidelity"); axis.grid(axis="y",alpha=.25); axis.legend(fontsize=8); fig.savefig(FIGURE,dpi=180); plt.close(fig); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
