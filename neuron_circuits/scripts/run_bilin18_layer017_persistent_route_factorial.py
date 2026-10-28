#!/usr/bin/env python3
"""Factor finite token edits into carried, repeated-embedding, and value-bus routes."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_router import normalized_embedding
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG=Path("configs/bilin18_layer017_persistent_route_factorial.json")
OUTPUT=TENSOR_ROOT/"layer017_persistent_route_factorial.json"
REPORT=TENSOR_ROOT/"LAYER017_PERSISTENT_ROUTE_FACTORIAL.md"


def median(values): return float(np.median(np.asarray(values,dtype=np.float64)))


def first_value_bus(model,x0):
    block=model.blocks[0]; mixed=block.lambdas[0]*x0+block.lambdas[1]*x0
    _,bus=block.attn(F.rms_norm(mixed,(model.config.n_embd,)),None)
    return bus


def hybrid_pres(model,path_x0,base_x0,base_bus,last_layer,position,dynamic_injection,dynamic_bus):
    x,endpoint_bus,rows=path_x0,None,[]
    for layer,block in enumerate(model.blocks[:last_layer+1]):
        injected=path_x0 if layer==0 or dynamic_injection else base_x0
        mixed=block.lambdas[0]*x+block.lambdas[1]*injected
        if layer==0:
            attention,endpoint_bus=block.attn(F.rms_norm(mixed,(model.config.n_embd,)),None)
        else:
            attention,_=block.attn(F.rms_norm(mixed,(model.config.n_embd,)),endpoint_bus if dynamic_bus else base_bus)
        pre=mixed+attention; rows.append(pre[:,position]); x=pre+block.mlp(F.rms_norm(pre,(model.config.n_embd,)))
    return torch.stack(rows,dim=1)


def product(block,pre,dimension):
    state=F.rms_norm(pre,(dimension,)); return block.mlp.Left(state)*block.mlp.Right(state)


def projection(target,value): return float((target*value).sum()/target.square().sum().clamp_min(1e-30))


@torch.no_grad()
def main():
    config=json.loads(CONFIG.read_text()); source=json.loads(Path(config["source_config"]).read_text()); data=json.loads(Path(source["data_config"]).read_text()); pairs=json.loads(Path(source["pair_config"]).read_text()); edit_position,final_position=data["edit_position"],data["causal_position"]
    model=load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval(); last=max(config["layers"])
    for module in (model.embedding,model.blocks,model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays={corpus:np.load(spec["path"],mmap_mode="r")[spec["start"]:spec["start"]+spec["rows"]] for corpus,spec in data["evaluation"].items()}; records={str(layer):[] for layer in config["layers"]}; max_error=0.; max_native_error=0.
    for corpus_index,(corpus,rows) in enumerate(arrays.items()):
        generator=np.random.default_rng(config["seed"]+corpus_index)
        for family,lexical_pairs in pairs["new_families"].items():
            for source_token,target_token in lexical_pairs:
                windows=select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],generator)
                for context_index,(row_index,start) in enumerate(windows):
                    raw=np.asarray(rows[row_index,start:start+data["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token; ids=torch.as_tensor(raw,device="cuda")[None]; changed_ids=torch.as_tensor(changed,device="cuda")[None]
                    base_native=forward_layers(model,ids,last_layer=last); changed_native=forward_layers(model,changed_ids,last_layer=last); base_x0=normalized_embedding(model,ids); path_x0=normalized_embedding(model,changed_ids); base_bus=first_value_bus(model,base_x0)
                    base_pres=hybrid_pres(model,base_x0,base_x0,base_bus,last,final_position,False,False)[0]
                    worlds={key:hybrid_pres(model,path_x0,base_x0,base_bus,last,final_position,bool(key[0]),bool(key[1]))[0] for key in itertools.product((0,1),repeat=2)}
                    for layer in config["layers"]:
                        block=model.blocks[layer]; dimension=model.config.n_embd; decoder=block.mlp.Down.weight.float(); base_product=product(block,base_pres[layer],dimension); deltas={key:product(block,pres[layer],dimension)-base_product for key,pres in worlds.items()}; full=deltas[(1,1)]
                        core=deltas[(0,0)]; injection=.5*((deltas[(1,0)]-deltas[(0,0)])+(deltas[(1,1)]-deltas[(0,1)])); bus=.5*((deltas[(0,1)]-deltas[(0,0)])+(deltas[(1,1)]-deltas[(1,0)])); error=float((core+injection+bus-full).norm()/full.norm().clamp_min(1e-30)); max_error=max(max_error,error)
                        native_base=F.rms_norm(base_native[layer]["pre"][0,final_position].float(),(dimension,)); native_changed=F.rms_norm(changed_native[layer]["pre"][0,final_position].float(),(dimension,)); native=block.mlp.Left(native_changed)*block.mlp.Right(native_changed)-block.mlp.Left(native_base)*block.mlp.Right(native_base); native_error=float((full-native).norm()/native.norm().clamp_min(1e-30)); max_native_error=max(max_native_error,native_error)
                        target_write=F.linear(full,decoder); terms={"core":core,"injection":injection,"bus":bus}; projections={name:projection(target_write,F.linear(value,decoder)) for name,value in terms.items()}; causal={"full":full,"core":deltas[(0,0)],"no_bus":deltas[(1,0)],"no_injection":deltas[(0,1)]}; effects=downstream_fidelities(model,layer,base_native[layer],causal,final_position)
                        records[str(layer)].append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","context":context_index,"projection":projections,"fidelity":{name:effects[name] for name in ("core","no_bus","no_injection")},"recomposition_error":error,"native_error":native_error})
                print("persistent factorial",corpus,family,source_token,target_token,flush=True)
    summaries={}
    for layer in config["layers"]:
        summaries[str(layer)]={}
        for corpus in arrays:
            selected=[row for row in records[str(layer)] if row["corpus"]==corpus]; summaries[str(layer)][corpus]={"projection":{name:median([row["projection"][name] for row in selected]) for name in ("core","injection","bus")},"fidelity":{name:median([row["fidelity"][name] for row in selected]) for name in ("core","no_bus","no_injection")}}
    row=lambda layer,corpus:summaries[str(layer)][corpus]
    hypotheses={
        "F0_route_factorial_is_exact":{"supported":max_error<=2e-5,"maximum_relative_error":max_error},
        "F1_manual_full_matches_native":{"supported":max_native_error<=2e-4,"maximum_relative_error":max_native_error},
        "F2_persistent_value_bus_is_material":{"supported":all(abs(row(l,c)["projection"]["bus"])>=.10 for l in range(4,18) for c in arrays)},
        "F3_repeated_embedding_is_material":{"supported":all(abs(row(l,c)["projection"]["injection"])>=.10 for l in range(4,18) for c in arrays)},
        "F4_carried_core_remains_material":{"supported":all(abs(row(l,c)["projection"]["core"])>=.20 for l in range(4,18) for c in arrays)},
        "F5_persistent_routes_explain_the_full_depth_stability":{"supported":all(row(l,c)["fidelity"]["core"]<=.80 for l in range(8,18) for c in arrays)},
        "F6_no_single_persistent_route_is_sufficient":{"supported":all(not all(row(l,c)["fidelity"][name]>=.90 for l in range(4,18) for c in arrays) for name in ("no_bus","no_injection"))},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"maximum_recomposition_error":max_error,"maximum_native_error":max_native_error,"summaries":summaries,"records":records},indent=2))
    lines=["# Persistent-route factorial through all 18 layers","","A finite source-token edit is run under four exact prefix coalitions: later repeated embedding injection frozen/dynamic × persistent Layer-0 value bus frozen/dynamic. Their product deltas define a core route (both frozen) plus two-factor Shapley terms for injection and bus. The full coalition is compared with the native checkpoint path; every coalition is patched through the complete suffix.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","| Layer | Corpus | Core/injection/bus local-write projection | Core/no-bus/no-injection downstream fidelity |","|---:|---|---:|---:|"]
    for l in config["layers"]:
        for c in arrays:
            item=row(l,c); p=item["projection"]; f=item["fidelity"]; lines.append(f"| {l} | {c} | {p['core']:.3f}/{p['injection']:.3f}/{p['bus']:.3f} | {f['core']:.3f}/{f['no_bus']:.3f}/{f['no_injection']:.3f} |")
    lines += ["",f"Maximum factorial recomposition error: **{max_error:.2e}**. Maximum manual-full versus native product error: **{max_native_error:.2e}**.","","`core` is not a no-embedding model: the token edit enters normally at Layer 0 and can propagate through carried residual/attention/MLP states. It freezes only later direct x0 reinjection and later reuse of the Layer-0 value bus. Coalition fidelity is relative to the complete final-position product edit, not total model behavior."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__=="__main__": main()
