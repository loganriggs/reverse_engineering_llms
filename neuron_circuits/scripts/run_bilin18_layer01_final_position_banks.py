#!/usr/bin/env python3
"""Prospective test of semantic banks defined only at the final query position."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_native_rotation_tournament import (
    make_rotation,
    rotate,
    selected_native,
    selected_rotated,
)
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer01_final_position_banks.json")
OUTPUT = TENSOR_ROOT / "layer01_final_position_banks.json"
REPORT = TENSOR_ROOT / "LAYER01_FINAL_POSITION_BANKS.md"
FIGURE = TENSOR_ROOT / "layer01_final_position_banks.png"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def final_coordinate_scores(delta, decoder, target_write, position):
    target = target_write[0, position].float()
    denominator = target.square().sum().clamp_min(1e-30)
    return delta[0, position].float() * (target @ decoder) / denominator


def sequence_coordinate_scores(delta, decoder, target_write, position):
    target = target_write[0, : position + 1].float()
    denominator = target.square().sum().clamp_min(1e-30)
    return (delta[0, : position + 1].float() * (target @ decoder)).sum(0) / denominator


def jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = set(left.tolist()), set(right.tolist())
    return len(a & b) / len(a | b)


@torch.inference_mode()
def evaluate_final_deltas(model, layer, base, state, deltas, position):
    """Compare routes with the complete final-position MLP change only."""
    block = model.blocks[layer]
    decoder = block.mlp.Down.weight.float()
    full_delta = torch.zeros_like(state["delta_product"])
    full_delta[:, position] = state["delta_product"][:, position]
    target_write = F.linear(full_delta[:, position].float(), decoder)
    local_denominator = target_write.square().sum().clamp_min(1e-30)
    names = list(deltas)
    products = []
    local = {}
    for name in names:
        delta = deltas[name]
        predicted = F.linear(delta[:, position].float(), decoder)
        local[name] = 1 - float((predicted - target_write).square().sum() / local_denominator)
        products.append(state["base_product"] + delta.to(state["base_product"].dtype))
    products.append(state["base_product"] + full_delta.to(state["base_product"].dtype))
    outputs = [base["output"]]
    x0s = [base["x0"]]
    buses = [base["value_bus"]]
    for product in products:
        outputs.append(base["pre"] + block.mlp.Down(product) + block.mlp.Down_bias)
        x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits = suffix_from_layer(model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), position)
    base_logits = logits[0]
    target_effect = logits[-1] - base_logits
    target_energy = fisher_inner(base_logits, target_effect, target_effect)
    rows = {}
    for index, name in enumerate(names):
        effect = logits[1 + index] - base_logits
        fidelity, cosine, _ = fisher_fidelity(base_logits, target_effect, effect)
        rows[name] = {
            "local_fidelity": local[name],
            "downstream_fidelity": fidelity,
            "downstream_cosine": cosine,
        }
    return rows, float(target_energy)


@torch.inference_mode()
def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG
    config = json.loads(config_path.read_text())
    output_path = Path(config.get("output_path", OUTPUT))
    report_path = Path(config.get("report_path", REPORT))
    figure_path = Path(config.get("figure_path", FIGURE))
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
    discovery_contexts = set(config.get("discovery_context_indices", [config["discovery_context_index"]]))
    discovery = [row for row in saved["semantic_records"] if row["corpus"] == config["discovery_corpus"] and row["context_index"] in discovery_contexts]
    family_names = list(parent["families"])
    results = {}

    def saved_trace(record, layer):
        spec = fixed["evaluation"][record["corpus"]]
        rows = arrays[record["corpus"]][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy(); changed[edit_position] = int(record["pair"].split("->")[1])
        return (
            forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)[layer],
            forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)[layer],
        )

    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        permutations = make_rotation(decoder.shape[1], config["rotation_rounds"], config["rotation_seed"] + layer, "cuda")
        native_final = {family: [] for family in family_names}
        native_sequence = {family: [] for family in family_names}
        rotated_final = {family: [] for family in family_names}
        all_final = []
        for record in discovery:
            base, counter = saved_trace(record, layer)
            state = channel_state(model, layer, base, counter, final_position)
            final_score = final_coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).cpu()
            sequence_score = sequence_coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).cpu()
            rotated_delta = rotate(state["delta_product"], permutations)
            rotated_decoder = rotate(decoder, permutations)
            rotated_score = final_coordinate_scores(rotated_delta, rotated_decoder, state["target_write"], final_position).cpu()
            native_final[record["family"]].append(final_score)
            native_sequence[record["family"]].append(sequence_score)
            rotated_final[record["family"]].append(rotated_score)
            all_final.append(final_score)
        banks = {}
        overlaps = {}
        for family in family_names:
            final_bank = top_indices(torch.stack(native_final[family]).abs().mean(0), config["rank"])
            sequence_bank = top_indices(torch.stack(native_sequence[family]).abs().mean(0), config["rank"])
            banks[family] = {
                "final": final_bank.cuda(),
                "sequence": sequence_bank.cuda(),
                "rotated": top_indices(torch.stack(rotated_final[family]).abs().mean(0), config["rank"]).cuda(),
            }
            overlaps[family] = jaccard(final_bank, sequence_bank)
        global_bank = top_indices(torch.stack(all_final).abs().mean(0), config["rank"]).cuda()
        records = []
        for corpus_index, (corpus, spec) in enumerate(fixed["evaluation"].items()):
            rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
            rng = np.random.default_rng(config["seed"] + corpus_index)
            for family, pairs in new_config["new_families"].items():
                for source_token, target_token in pairs:
                    windows = select_windows(rows, source_token, edit_position, final_position, config["contexts_per_pair"], rng)
                    for context_index, (row_index, start) in enumerate(windows):
                        raw = np.asarray(rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64).copy()
                        changed = raw.copy(); changed[edit_position] = target_token
                        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)[layer]
                        counter = forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)[layer]
                        state = channel_state(model, layer, base, counter, final_position)
                        final_delta = torch.zeros_like(state["delta_product"])
                        final_delta[:, final_position] = state["delta_product"][:, final_position]
                        oracle_score = final_coordinate_scores(final_delta, decoder, state["target_write"], final_position).abs()
                        deltas = {
                            "global": selected_native(final_delta, global_bank),
                            "sequence": selected_native(final_delta, banks[family]["sequence"]),
                            "oracle": selected_native(final_delta, top_indices(oracle_score, config["rank"])),
                        }
                        for candidate in family_names:
                            deltas[f"family_{candidate}"] = selected_native(final_delta, banks[candidate]["final"])
                        deltas["rotated"] = selected_rotated(final_delta, banks[family]["rotated"], permutations)
                        effects, target_energy = evaluate_final_deltas(model, layer, base, state, deltas, final_position)
                        correct = effects[f"family_{family}"]["downstream_fidelity"]
                        wrong = [effects[f"family_{name}"]["downstream_fidelity"] for name in family_names if name != family]
                        records.append({
                            "corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "context": context_index,
                            "correct": correct, "wrong": median(wrong), "advantage": correct - median(wrong),
                            "correct_local": effects[f"family_{family}"]["local_fidelity"],
                            "global": effects["global"]["downstream_fidelity"],
                            "sequence": effects["sequence"]["downstream_fidelity"],
                            "rotated": effects["rotated"]["downstream_fidelity"],
                            "oracle": effects["oracle"]["downstream_fidelity"],
                            "target_energy": target_energy,
                            "family_effects": {name: effects[f"family_{name}"]["downstream_fidelity"] for name in family_names},
                        })
                    print("final bank", layer, corpus, family, source_token, target_token, flush=True)
        summaries = {}
        for corpus in fixed["evaluation"]:
            selected = [row for row in records if row["corpus"] == corpus]
            summaries[corpus] = {key: median([row[key] for row in selected]) for key in ("correct","wrong","advantage","correct_local","global","sequence","rotated","oracle","target_energy")}
            summaries[corpus]["win_rate"] = float(np.mean([row["advantage"] > 0 for row in selected]))
        results[str(layer)] = {
            "overlaps": overlaps,
            "bank_indices": {
                family:{kind:value.detach().cpu().tolist() for kind,value in bank.items()}
                for family,bank in banks.items()
            },
            "summaries": summaries,
            "records": records,
        }

    def row(layer, corpus): return results[str(layer)]["summaries"][corpus]
    hypotheses = {
        "F0_final_bank_transfers":{"supported":all(row(l,c)["advantage"] >= .03 for l in config["layers"] for c in fixed["evaluation"])},
        "F1_correct_family_wins":{"supported":all(row(l,c)["win_rate"] >= .65 for l in config["layers"] for c in fixed["evaluation"])},
        "F2_final_bank_beats_sequence_bank":{"supported":all(row(l,c)["correct"] - row(l,c)["sequence"] >= .03 for l in config["layers"] for c in fixed["evaluation"])},
        "F3_native_basis_is_privileged":{"supported":all(row(l,c)["correct"] - row(l,c)["rotated"] >= .05 for l in config["layers"] for c in fixed["evaluation"])},
        "F4_oracle_gap_remains":{"supported":all(row(l,c)["oracle"] - row(l,c)["correct"] >= .10 for l in config["layers"] for c in fixed["evaluation"])},
        "F5_ood_retention":{"supported":all(row(l,"tinystories")["correct"] >= .75 * row(l,"fineweb")["correct"] for l in config["layers"])},
        "F6_scopes_share_nonrandom_but_distinct_channels":{"supported":all(.05 <= median(list(results[str(l)]["overlaps"].values())) < .50 for l in config["layers"])},
        "F7_final_patch_has_measurable_effect":{"supported":all(row(l,c)["target_energy"] >= 1e-8 for l in config["layers"] for c in fixed["evaluation"])},
    }
    output_path.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2))
    lines=["# Prospective final-position semantic banks in Layers 0 and 1","",f"Banks are frozen on the original nineteen substitutions using only each MLP's write at the final query position and discovery contexts `{sorted(discovery_contexts)}`. Seventeen declared new substitutions, three contexts per pair, are evaluated on FineWeb and TinyStories. The causal target is the complete final-position MLP change with all other MLP positions held at the base run.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","![Final-position banks](layer01_final_position_banks.png)","","| Layer | Corpus | Correct / wrong / advantage (win) | Sequence bank | Global | Rotated | Oracle | Local correct | Target Fisher energy |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus); lines.append(f"| {layer} | {corpus} | {item['correct']:.3f} / {item['wrong']:.3f} / {item['advantage']:.3f} ({item['win_rate']:.2f}) | {item['sequence']:.3f} | {item['global']:.3f} | {item['rotated']:.3f} | {item['oracle']:.3f} | {item['correct_local']:.3f} | {item['target_energy']:.2e} |")
    lines += ["","Final/sequence bank Jaccard by family:",""]
    for layer in config["layers"]: lines.append(f"- Layer {layer}: " + ", ".join(f"{family}={value:.3f}" for family,value in results[str(layer)]["overlaps"].items()))
    lines += ["","This separates a position-local feature claim from the earlier distributed-sequence circuit claim. High oracle but weak family transfer would mean native channels are causally useful without supporting reusable semantic banks at the query position."]
    report_path.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); cells=[(0,"fineweb"),(0,"tinystories"),(1,"fineweb"),(1,"tinystories")]; x=np.arange(4); labels=["L0 Fine","L0 Tiny","L1 Fine","L1 Tiny"]
    for offset,name in zip((-.3,-.1,.1,.3),("correct","sequence","rotated","oracle")): axes[0].bar(x+offset,[row(l,c)[name] for l,c in cells],.2,label=name)
    axes[0].set_xticks(x,labels,rotation=20); axes[0].set_ylabel("downstream fidelity"); axes[0].legend(fontsize=8); axes[0].grid(axis="y",alpha=.25)
    axes[1].bar(x,[row(l,c)["advantage"] for l,c in cells]); axes[1].axhline(0,color="black",lw=.8); axes[1].set_xticks(x,labels,rotation=20); axes[1].set_ylabel("correct minus wrong-family"); axes[1].grid(axis="y",alpha=.25)
    fig.savefig(figure_path,dpi=180); plt.close(fig)
    print(json.dumps(hypotheses,indent=2)); print(report_path)


if __name__ == "__main__": main()
