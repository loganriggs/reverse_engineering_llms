#!/usr/bin/env python3
"""Prospective validation of frozen Layer-0/1 native banks on new lexical pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_family_bank_structure import recurrence_ranking
from run_bilin18_layer01_native_rotation_tournament import (
    coordinate_scores,
    evaluate_deltas,
    make_rotation,
    rotate,
    selected_native,
    selected_rotated,
)
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_new_pair_validation.json")
OUTPUT = TENSOR_ROOT / "layer01_new_pair_validation.json"
REPORT = TENSOR_ROOT / "LAYER01_NEW_PAIR_VALIDATION.md"
FIGURE = TENSOR_ROOT / "layer01_new_pair_validation.png"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    core_config = json.loads(Path(config["source_config"]).read_text())
    bank_config = json.loads(Path(core_config["source_config"]).read_text())
    frontier_config = json.loads(Path(bank_config["source_config"]).read_text())
    red_config = json.loads(Path(frontier_config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    new_config = json.loads(Path(config["new_pair_config"]).read_text())
    saved = json.loads(Path(config["source_results"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    rank = config["rank"]
    family_names = list(parent["families"])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays = {corpus: np.load(spec["path"], mmap_mode="r") for corpus, spec in fixed["evaluation"].items()}
    specs = fixed["evaluation"]

    def reconstruct_saved(record, layer):
        spec = specs[record["corpus"]]
        rows = arrays[record["corpus"]][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy(); changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counter = forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)
        return base[layer], counter[layer]

    discovery_saved = [r for r in saved["semantic_records"] if r["corpus"] == config["discovery_corpus"] and r["context_index"] == config["discovery_context_index"]]
    results = {}
    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        permutations = make_rotation(decoder.shape[1], config["rotation_rounds"], config["rotation_seed"], "cuda")
        rotated_decoder = rotate(decoder, permutations)
        native_by_family = {family: [] for family in family_names}
        rotated_by_family = {family: [] for family in family_names}
        all_native = []
        for record in discovery_saved:
            base, counter = reconstruct_saved(record, layer)
            state = channel_state(model, layer, base, counter, final_position)
            native = coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).cpu()
            rotated_coordinates = rotate(state["delta_product"], permutations)
            rotated_score = coordinate_scores(rotated_coordinates, rotated_decoder, state["target_write"], final_position).cpu()
            native_by_family[record["family"]].append(native)
            rotated_by_family[record["family"]].append(rotated_score)
            all_native.append(native)
        banks = {}
        for family in family_names:
            native_rows = torch.stack(native_by_family[family])
            rotated_rows = torch.stack(rotated_by_family[family])
            _, frequency = recurrence_ranking(native_rows, rank)
            banks[family] = {
                "magnitude": top_indices(native_rows.abs().mean(0), rank).cuda(),
                "signed": top_indices(native_rows.mean(0).abs(), rank).cuda(),
                "strict": torch.nonzero(frequency >= 2, as_tuple=True)[0].cuda(),
                "rotated": top_indices(rotated_rows.abs().mean(0), rank).cuda(),
            }
        global_bank = top_indices(torch.stack(all_native).abs().mean(0), rank).cuda()

        layer_records = []
        for corpus_index, (corpus, corpus_spec) in enumerate(specs.items()):
            source = arrays[corpus]
            rows = source[corpus_spec["start"]:corpus_spec["start"] + corpus_spec["rows"]]
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
                        oracle_score = coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).abs()
                        deltas = {
                            "global": selected_native(state["delta_product"], global_bank),
                            "oracle": selected_native(state["delta_product"], top_indices(oracle_score, rank)),
                        }
                        for candidate in family_names:
                            deltas[f"family_{candidate}"] = selected_native(state["delta_product"], banks[candidate]["magnitude"])
                            deltas[f"rotated_{candidate}"] = selected_rotated(state["delta_product"], banks[candidate]["rotated"], permutations)
                        deltas["correct_signed"] = selected_native(state["delta_product"], banks[family]["signed"])
                        deltas["correct_strict"] = selected_native(state["delta_product"], banks[family]["strict"])
                        effects = evaluate_deltas(model, layer, base, counter, state, deltas, final_position)
                        correct = effects[f"family_{family}"]["downstream_fidelity"]
                        wrong = [effects[f"family_{name}"]["downstream_fidelity"] for name in family_names if name != family]
                        rotated_correct = effects[f"rotated_{family}"]["downstream_fidelity"]
                        rotated_wrong = [effects[f"rotated_{name}"]["downstream_fidelity"] for name in family_names if name != family]
                        layer_records.append({
                            "corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}",
                            "context": context_index, "row": int(row_index),
                            "correct": correct, "wrong_median": median(wrong), "advantage": correct - median(wrong),
                            "global": effects["global"]["downstream_fidelity"],
                            "oracle": effects["oracle"]["downstream_fidelity"],
                            "signed": effects["correct_signed"]["downstream_fidelity"],
                            "strict": effects["correct_strict"]["downstream_fidelity"],
                            "strict_size": len(banks[family]["strict"]),
                            "rotated_correct": rotated_correct,
                            "rotated_wrong_median": median(rotated_wrong),
                            "family_effects": {
                                name: effects[f"family_{name}"]["downstream_fidelity"]
                                for name in family_names
                            },
                            "rotated_family_effects": {
                                name: effects[f"rotated_{name}"]["downstream_fidelity"]
                                for name in family_names
                            },
                        })
                    print("new pair", layer, corpus, family, source_token, target_token, flush=True)
        summaries = {}
        for corpus in fixed["evaluation"]:
            selected = [row for row in layer_records if row["corpus"] == corpus]
            summary = {key: median([row[key] for row in selected]) for key in ("correct","wrong_median","advantage","global","oracle","signed","strict","strict_size","rotated_correct","rotated_wrong_median")}
            summary["win_rate"] = float(np.mean([row["advantage"] > 0 for row in selected]))
            families = {}
            for family in family_names:
                subset = [row for row in selected if row["family"] == family]
                families[family] = {"correct": median([row["correct"] for row in subset]), "advantage": median([row["advantage"] for row in subset]), "win_rate": float(np.mean([row["advantage"] > 0 for row in subset])), "strict": median([row["strict"] for row in subset])}
            summaries[corpus] = {"summary": summary, "families": families}
        results[str(layer)] = {"summaries": summaries, "records": layer_records, "strict_sizes": {family: len(bank["strict"]) for family, bank in banks.items()}}

    def row(layer, corpus): return results[str(layer)]["summaries"][corpus]["summary"]
    hypotheses = {
        "P0_native_family_banks_transfer_to_new_pairs": {"supported": all(row(l,c)["advantage"] >= .03 for l in config["layers"] for c in fixed["evaluation"])},
        "P1_correct_family_wins_most_examples": {"supported": all(row(l,c)["win_rate"] >= .65 for l in config["layers"] for c in fixed["evaluation"])},
        "P2_layer1_family_bank_beats_global": {"supported": all(row(1,c)["correct"] - row(1,c)["global"] >= .03 for c in fixed["evaluation"])},
        "P3_layer0_is_primarily_generic": {"supported": all(row(0,c)["correct"] - row(0,c)["global"] <= .03 for c in fixed["evaluation"])},
        "P4_strict_recurrent_core_transfers": {"supported": all(row(l,c)["strict"] >= .05 for l in config["layers"] for c in fixed["evaluation"])},
        "P5_strict_core_retains_half_of_family_bank": {"supported": all(row(l,c)["strict"] >= .5 * row(l,c)["correct"] for l in config["layers"] for c in fixed["evaluation"])},
        "P6_signed_common_bank_matches_magnitude": {"supported": all(row(l,c)["correct"] - row(l,c)["signed"] <= .05 for l in config["layers"] for c in fixed["evaluation"])},
        "P7_native_axis_advantage_transfers": {"supported": all(row(l,c)["correct"] - row(l,c)["rotated_correct"] >= .05 for l in config["layers"] for c in fixed["evaluation"])},
        "P8_oracle_gap_persists": {"supported": all(row(l,c)["oracle"] >= max(row(l,c)["correct"],row(l,c)["global"]) + .10 for l in config["layers"] for c in fixed["evaluation"])},
        "P9_new_pair_effect_transfers_ood": {"supported": all(row(l,"tinystories")["correct"] >= .75 * row(l,"fineweb")["correct"] for l in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2))
    lines=["# Prospective Layer-0/1 native-bank transfer to seventeen new lexical pairs","","All native, signed, strict-recurrent, global, and rotated banks are frozen from the original nineteen substitutions. They are evaluated on seventeen previously declared new substitutions, three independently selected contexts per pair, on FineWeb and TinyStories. No new-pair effect enters bank selection.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","![New-pair validation](layer01_new_pair_validation.png)","","| Layer | Corpus | Correct | Wrong | Advantage/win | Global | Signed | Strict (size) | Rotated correct | Oracle |","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            item=row(layer,corpus)
            lines.append(f"| {layer} | {corpus} | {item['correct']:.3f} | {item['wrong_median']:.3f} | {item['advantage']:.3f}/{item['win_rate']:.2f} | {item['global']:.3f} | {item['signed']:.3f} | {item['strict']:.3f} ({item['strict_size']:.0f}) | {item['rotated_correct']:.3f} | {item['oracle']:.3f} |")
    lines += ["","Correct-versus-wrong tests semantic family specificity. Correct-versus-global tests whether the family label improves on a generic lexical-edit bank. Native-versus-rotated tests coordinate privilege. Strict-core fidelity tests the small shared kernel; oracle distance measures the remaining context-specific tail."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained"); x=np.arange(4); labels=["L0 Fine","L0 Tiny","L1 Fine","L1 Tiny"]; cells=[(0,"fineweb"),(0,"tinystories"),(1,"fineweb"),(1,"tinystories")]
    width=.2
    for off,name in zip((-.3,-.1,.1,.3),("correct","global","strict","rotated_correct")):
        axes[0].bar(x+off,[row(l,c)[name] for l,c in cells],width,label=name)
    axes[0].set_xticks(x,labels,rotation=20); axes[0].set_ylabel("downstream fidelity"); axes[0].legend(fontsize=8); axes[0].grid(axis="y",alpha=.25)
    axes[1].bar(x,[row(l,c)["advantage"] for l,c in cells]); axes[1].axhline(0,color="black",linewidth=.8); axes[1].set_xticks(x,labels,rotation=20); axes[1].set_ylabel("correct minus wrong-family median"); axes[1].grid(axis="y",alpha=.25)
    fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps(hypotheses,indent=2),flush=True); print(REPORT,flush=True)


if __name__ == "__main__": main()
