#!/usr/bin/env python3
"""Exact pre-MLP source factorial for prospective Layer-0/1 semantic banks."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_family_bank_structure import recurrence_ranking
from run_bilin18_layer01_native_rotation_tournament import coordinate_scores
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_semantic_source_factorial.json")
OUTPUT = TENSOR_ROOT / "layer01_semantic_source_factorial.json"
REPORT = TENSOR_ROOT / "LAYER01_SEMANTIC_SOURCE_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "layer01_semantic_source_factorial.png"


def coalition_keys(source_count: int) -> list[tuple[int, ...]]:
    return list(itertools.product((0, 1), repeat=source_count))


def shapley(outputs: dict[tuple[int, ...], torch.Tensor]) -> list[torch.Tensor]:
    source_count = len(next(iter(outputs)))
    result = []
    for source in range(source_count):
        value = torch.zeros_like(next(iter(outputs.values())))
        others = [index for index in range(source_count) if index != source]
        for size in range(source_count):
            weight = math.factorial(size) * math.factorial(source_count - size - 1) / math.factorial(source_count)
            for subset in itertools.combinations(others, size):
                absent = [0] * source_count
                for index in subset: absent[index] = 1
                present = absent.copy(); present[source] = 1
                value = value + weight * (outputs[tuple(present)] - outputs[tuple(absent)])
        result.append(value)
    return result


def fidelity(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return 1 - float((prediction - target).square().sum() / target.square().sum().clamp_min(1e-30))


def attribution(target: torch.Tensor, value: torch.Tensor) -> tuple[float, float]:
    denominator = target.square().sum().clamp_min(1e-30)
    return float((target * value).sum() / denominator), float(value.square().sum() / denominator)


def write_at_position(
    delta_product: torch.Tensor,
    decoder: torch.Tensor,
    position: int,
    selected: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode a product-channel change only at the declared causal position.

    ``delta_product`` contains every sequence position because the source
    factorial is constructed at the complete block boundary.  Semantic banks,
    however, were discovered and causally evaluated at ``position``.  Decoding
    the complete tensor would mostly measure the directly edited token one
    position earlier and would therefore answer a different question.
    """
    value = delta_product[..., position, :]
    if selected is not None:
        masked = torch.zeros_like(value)
        masked[..., selected] = value[..., selected]
        value = masked
    return F.linear(value, decoder)


def write_through_position(
    delta_product: torch.Tensor,
    decoder: torch.Tensor,
    position: int,
    selected: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode the distributed sequence write through the prediction position."""
    value = delta_product[..., : position + 1, :]
    if selected is not None:
        masked = torch.zeros_like(value)
        masked[..., selected] = value[..., selected]
        value = masked
    return F.linear(value, decoder)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    validation_config = json.loads(Path(config["source_config"]).read_text())
    core_config = json.loads(Path(validation_config["source_config"]).read_text())
    bank_config = json.loads(Path(core_config["source_config"]).read_text())
    frontier_config = json.loads(Path(bank_config["source_config"]).read_text())
    red_config = json.loads(Path(frontier_config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    new_config = json.loads(Path(config["new_pair_config"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays = {corpus: np.load(spec["path"], mmap_mode="r") for corpus, spec in fixed["evaluation"].items()}
    specs = fixed["evaluation"]
    discovery_saved = [r for r in saved["semantic_records"] if r["corpus"] == config["discovery_corpus"] and r["context_index"] == config["discovery_context_index"]]
    results = {}
    maximum_error = 0.0

    def trace_saved(record, layer):
        spec=specs[record["corpus"]]; rows=arrays[record["corpus"]][spec["start"]:spec["start"]+spec["rows"]]
        raw=np.asarray(rows[record["row"],record["source_start"]:record["source_start"]+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=int(record["pair"].split("->")[1])
        return forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer),forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)

    for layer in config["layers"]:
        block=model.blocks[layer]; decoder=block.mlp.Down.weight.float(); family_rows={family:[] for family in parent["families"]}
        for record in discovery_saved:
            base_trace,counter_trace=trace_saved(record,layer); state=channel_state(model,layer,base_trace[layer],counter_trace[layer],final_position)
            family_rows[record["family"]].append(coordinate_scores(state["delta_product"],decoder,state["target_write"],final_position).cpu())
        banks={}
        for family,rows in family_rows.items():
            matrix=torch.stack(rows); _,frequency=recurrence_ranking(matrix,config["rank"])
            banks[family]={"family":top_indices(matrix.abs().mean(0),config["rank"]).cuda(),"strict":torch.nonzero(frequency>=2,as_tuple=True)[0].cuda()}
        source_names=("embedding","attention") if layer==0 else ("carried","injection","attention")
        records=[]
        for corpus_index,(corpus,spec) in enumerate(specs.items()):
            rows=arrays[corpus][spec["start"]:spec["start"]+spec["rows"]]; rng=np.random.default_rng(config["seed"]+corpus_index)
            for family,pairs in new_config["new_families"].items():
                for source_token,target_token in pairs:
                    windows=select_windows(rows,source_token,edit_position,final_position,config["contexts_per_pair"],rng)
                    for context_index,(row_index,start) in enumerate(windows):
                        raw=np.asarray(rows[row_index,start:start+fixed["sequence_length"]],dtype=np.int64).copy(); changed=raw.copy(); changed[edit_position]=target_token
                        base_trace=forward_layers(model,torch.as_tensor(raw,device="cuda")[None],last_layer=layer); counter_trace=forward_layers(model,torch.as_tensor(changed,device="cuda")[None],last_layer=layer)
                        base,counter=base_trace[layer],counter_trace[layer]
                        if layer==0:
                            components=[counter["mixed"].float()-base["mixed"].float(),counter["attention_write"].float()-base["attention_write"].float()]
                        else:
                            components=[
                                block.lambdas[0].float()*(counter_trace[layer-1]["output"].float()-base_trace[layer-1]["output"].float()),
                                block.lambdas[1].float()*(counter["x0"].float()-base["x0"].float()),
                                counter["attention_write"].float()-base["attention_write"].float(),
                            ]
                        pre_error=float((sum(components)- (counter["pre"].float()-base["pre"].float())).norm()/(counter["pre"].float()-base["pre"].float()).norm().clamp_min(1e-30)); maximum_error=max(maximum_error,pre_error)
                        dimension=base["pre"].shape[-1]; base_input=F.rms_norm(base["pre"].float(),(dimension,)); base_product=block.mlp.Left(base_input)*block.mlp.Right(base_input)
                        outputs={
                            scope:{kind:{} for kind in ("full","family","strict")}
                            for scope in ("sequence","final")
                        }
                        for key in coalition_keys(len(components)):
                            pre=base["pre"].float()+sum((component for bit,component in zip(key,components) if bit),torch.zeros_like(base["pre"].float()))
                            state=F.rms_norm(pre,(dimension,)); product=block.mlp.Left(state)*block.mlp.Right(state); delta=product-base_product
                            outputs["sequence"]["full"][key]=write_through_position(delta,decoder,final_position)
                            outputs["final"]["full"][key]=write_at_position(delta,decoder,final_position)
                            for kind in ("family","strict"):
                                outputs["sequence"][kind][key]=write_through_position(
                                    delta,decoder,final_position,banks[family][kind]
                                )
                                outputs["final"][kind][key]=write_at_position(
                                    delta,decoder,final_position,banks[family][kind]
                                )
                        metrics={scope:{} for scope in outputs}
                        for scope,scope_outputs in outputs.items():
                            for kind,coalitions in scope_outputs.items():
                                target=coalitions[(1,)*len(components)]-coalitions[(0,)*len(components)]; phis=shapley(coalitions); recomposed=sum(phis)
                                error=float((recomposed-target).norm()/target.norm().clamp_min(1e-30)); maximum_error=max(maximum_error,error)
                                metrics[scope][kind]={}
                                for index,name in enumerate(source_names):
                                    projection,energy=attribution(target,phis[index]); singleton=tuple(int(i==index) for i in range(len(components)))
                                    metrics[scope][kind][name]={"projection":projection,"energy":energy,"singleton_fidelity":fidelity(target,coalitions[singleton]-coalitions[(0,)*len(components)])}
                        records.append({"corpus":corpus,"family":family,"pair":f"{source_token}->{target_token}","context":context_index,"metrics":metrics})
                    print("source factorial",layer,corpus,family,source_token,target_token,flush=True)
        summaries={}
        for corpus in fixed["evaluation"]:
            selected=[r for r in records if r["corpus"]==corpus]
            summaries[corpus]={scope:{kind:{source:{metric:float(np.median([r["metrics"][scope][kind][source][metric] for r in selected])) for metric in ("projection","energy","singleton_fidelity")} for source in source_names} for kind in ("full","family","strict")} for scope in ("sequence","final")}
        results[str(layer)]={"source_names":source_names,"summaries":summaries,"records":records}

    def projection(layer,corpus,kind,source,scope="sequence"): return results[str(layer)]["summaries"][corpus][scope][kind][source]["projection"]
    hypotheses={
        "A0_boundary_factorial_is_exact":{"supported":maximum_error<=2e-5,"maximum_relative_error":maximum_error},
        "A1_layer0_family_bank_is_embedding_dominated":{"supported":all(projection(0,c,"family","embedding")>=.6 for c in fixed["evaluation"])},
        "A2_layer0_attention_is_material":{"supported":all(projection(0,c,"family","attention")>=.15 for c in fixed["evaluation"])},
        "A3_layer1_family_bank_is_carried_state_dominated":{"supported":all(projection(1,c,"family","carried")>=.5 and projection(1,c,"family","carried")==max(projection(1,c,"family",s) for s in ("carried","injection","attention")) for c in fixed["evaluation"])},
        "A4_layer1_direct_injection_is_a_minor_source":{"supported":all(projection(1,c,"family","injection")<=.25 for c in fixed["evaluation"])},
        "A5_layer1_attention_is_material":{"supported":all(projection(1,c,"family","attention")>=.1 for c in fixed["evaluation"])},
        "A6_strict_core_is_more_source_concentrated":{"supported":all(max(results[str(l)]["summaries"][c]["sequence"]["strict"][s]["projection"] for s in results[str(l)]["source_names"])>=max(results[str(l)]["summaries"][c]["sequence"]["family"][s]["projection"] for s in results[str(l)]["source_names"])+.05 for l in config["layers"] for c in fixed["evaluation"])},
        "A7_source_attribution_transfers_ood":{"supported":all(abs(results[str(l)]["summaries"]["fineweb"]["sequence"]["family"][s]["projection"]-results[str(l)]["summaries"]["tinystories"]["sequence"]["family"][s]["projection"])<=.15 for l in config["layers"] for s in results[str(l)]["source_names"])},
        "A8_no_single_boundary_source_is_sufficient":{"supported":all(max(results[str(l)]["summaries"][c]["sequence"]["family"][s]["singleton_fidelity"] for s in results[str(l)]["source_names"])<.9 for l in config["layers"] for c in fixed["evaluation"])},
    }
    correction={
        "primary_metric_position":int(final_position),
        "edited_token_position":int(edit_position),
        "note":"The preregistered primary scope is the distributed write through the prediction position, matching family-bank discovery and patching. A final-position-only diagnostic distinguishes where the edit is stored from where it is read.",
    }
    OUTPUT.write_text(json.dumps({"config":config,"correction":correction,"hypotheses":hypotheses,"maximum_relative_error":maximum_error,"layers":results},indent=2))
    lines=["# Exact boundary-source factorial for prospective Layer-0/1 semantic banks","","At each MLP boundary the counterfactual pre-MLP change is decomposed additively. Layer 0 uses embedding-mix and local-attention changes; Layer 1 uses carried Layer-0 output, direct repeated-embedding injection, and local attention. Every coalition is passed through RMSNorm and the native bilinear MLP. Shapley values attribute the complete write, rank-256 family bank, and strict recurrent core.","","## Two legitimate spatial scopes","",f"The preregistered primary scope decodes the distributed write through prediction position `{final_position}`, matching how the family banks were discovered and patched. The diagnostic scope decodes only position `{final_position}`. The substituted token is at position `{edit_position}`. These answer different questions: **where the edit is represented across the sequence** versus **what reaches the query position**.","", "At the final position, the direct embedding-delta terms are exactly zero by causal layout: that token's identity was not changed. Consequently, Layer-0 attention sufficiency and zero direct-injection attribution there are architectural controls, not discoveries that embeddings are globally unimportant. Attention is itself an endogenous function of embedding/current states and QKOV weights.","","## Locked hypotheses (distributed-sequence scope)",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","![Semantic source factorial](layer01_semantic_source_factorial.png)","","| Scope | Layer | Corpus | Target | Source | Shapley projection | Energy ratio | Source-alone fidelity |","|---|---:|---|---|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            for scope in ("sequence","final"):
                for kind in ("full","family","strict"):
                    for source,row in results[str(layer)]["summaries"][corpus][scope][kind].items(): lines.append(f"| {scope} | {layer} | {corpus} | {kind} | {source} | {row['projection']:.3f} | {row['energy']:.3f} | {row['singleton_fidelity']:.3f} |")
    lines += ["",f"Maximum source/Shapley recomposition error: **{maximum_error:.3e}**.","","This is an exact finite factorial at the declared MLP boundary. The attention vector is treated as one additive boundary source; the experiment does not claim that attention is independent of the embedding or carried state upstream. Source projection allocates the target write, while source-alone fidelity asks whether the corresponding hybrid boundary state is sufficient."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for row_index,scope in enumerate(("sequence","final")):
      for axis,layer in zip(axes[row_index],(0,1)):
        names=results[str(layer)]["source_names"]; x=np.arange(len(names)); width=.25
        for offset,(corpus,color) in zip((-.125,.125),(("fineweb","tab:blue"),("tinystories","tab:orange"))): axis.bar(x+offset,[results[str(layer)]["summaries"][corpus][scope]["family"][s]["projection"] for s in names],width,label=corpus,color=color)
        axis.set_xticks(x,names); axis.set_ylabel("family-bank Shapley projection"); axis.set_title(f"{scope}: Layer {layer}"); axis.axhline(0,color="black",linewidth=.7); axis.legend(fontsize=8); axis.grid(axis="y",alpha=.25)
    fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps(hypotheses,indent=2),flush=True); print(REPORT,flush=True)


if __name__=="__main__": main()
