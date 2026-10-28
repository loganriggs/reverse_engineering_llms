#!/usr/bin/env python3
"""Finite test of channels recurring in multiple semantic-family edit supports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_family_bank_structure import recurrence_ranking
from run_bilin18_layer01_native_rotation_tournament import coordinate_scores, evaluate_deltas, selected_native, summarize
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_strict_recurrent_core.json")
OUTPUT = TENSOR_ROOT / "layer01_strict_recurrent_core.json"
REPORT = TENSOR_ROOT / "LAYER01_STRICT_RECURRENT_CORE.md"
FIGURE = TENSOR_ROOT / "layer01_strict_recurrent_core.png"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bank_config = json.loads(Path(config["source_config"]).read_text())
    frontier_config = json.loads(Path(bank_config["source_config"]).read_text())
    red_config = json.loads(Path(frontier_config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, _ = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays = {corpus: np.load(spec["path"], mmap_mode="r") for corpus, spec in fixed["evaluation"].items()}
    specs = fixed["evaluation"]

    def reconstruct(record, layer):
        spec = specs[record["corpus"]]
        rows = arrays[record["corpus"]][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy(); changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)
        return base[layer], counterfactual[layer]

    discovery_saved = [r for r in saved["semantic_records"] if r["corpus"] == config["discovery_corpus"] and r["context_index"] == config["discovery_context_index"]]
    query_saved = [r for r in saved["semantic_records"] if r["context_index"] == config["evaluation_context_index"]]
    family_by_pair = {r["pair"]: r["family"] for r in discovery_saved}
    results = {}
    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        signed = {}
        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            signed[record["pair"]] = coordinate_scores(state["delta_product"], decoder, state["target_write"], final_position).cpu()
        cache, sizes = {}, []
        for pair in signed:
            members = [p for p in signed if p != pair and family_by_pair[p] == family_by_pair[pair]]
            rows = torch.stack([signed[p] for p in members])
            _, frequency = recurrence_ranking(rows, config["reference_rank"])
            strict = torch.nonzero(frequency >= 2, as_tuple=True)[0]
            size = len(strict)
            sizes.append(size)
            magnitude = top_indices(rows.abs().mean(0), size)
            signed_common = top_indices(rows.mean(0).abs(), size)
            magnitude256 = top_indices(rows.abs().mean(0), config["reference_rank"])
            cache[pair] = {"strict": strict.cuda(), "matched_magnitude": magnitude.cuda(), "matched_signed": signed_common.cuda(), "magnitude256": magnitude256.cuda()}
        records, route_names = [], []
        for record in query_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            deltas = {name: selected_native(state["delta_product"], indices) for name, indices in cache[record["pair"]].items()}
            if not route_names: route_names = list(deltas)
            routes = evaluate_deltas(model, layer, base, counterfactual, state, deltas, final_position)
            records.append({"corpus": record["corpus"], "family": record["family"], "pair": record["pair"], "strict_size": len(cache[record["pair"]]["strict"]), "routes": routes})
        summaries = {corpus: summarize([r for r in records if r["corpus"] == corpus], route_names) for corpus in fixed["evaluation"]}
        results[str(layer)] = {"median_size": float(np.median(sizes)), "minimum_size": min(sizes), "maximum_size": max(sizes), "summaries": summaries, "records": records}

    def value(layer, corpus, route): return results[str(layer)]["summaries"][corpus][route]["downstream_fidelity"]
    hypotheses = {
        "S1_strict_core_captures_half_of_bank_effect": {"supported": all(value(l,c,"strict") >= .5 * value(l,c,"magnitude256") for l in config["layers"] for c in fixed["evaluation"])},
        "S2_recurrence_beats_equal_size_magnitude": {"supported": all(value(l,c,"strict") - value(l,c,"matched_magnitude") >= .03 for l in config["layers"] for c in fixed["evaluation"])},
        "S3_strict_core_has_positive_ood_effect": {"supported": all(value(l,"tinystories","strict") >= .05 for l in config["layers"])},
        "S4_layer1_strict_core_is_stronger": {"supported": all(value(1,c,"strict") - value(0,c,"strict") >= .03 for c in fixed["evaluation"])},
        "S5_strict_core_is_small": {"supported": all(results[str(l)]["median_size"] <= 96 for l in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"layers":results},indent=2))
    lines=["# Strict recurrent semantic cores in Layers 0–1","","Only channels appearing in the top-256 contribution sets of at least two leave-one-pair-out discovery members are retained. Their variable-size finite effect is compared with equal-size magnitude and signed-mean banks, plus the complete 256-channel magnitude bank.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","![Strict recurrent cores](layer01_strict_recurrent_core.png)","","| Layer | Core size median/min/max | Corpus | Strict recurrent | Equal-size magnitude | Equal-size signed | Magnitude-256 |","|---:|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        size=results[str(layer)]
        for corpus in fixed["evaluation"]:
            lines.append(f"| {layer} | {size['median_size']:.0f}/{size['minimum_size']}/{size['maximum_size']} | {corpus} | {value(layer,corpus,'strict'):.3f} | {value(layer,corpus,'matched_magnitude'):.3f} | {value(layer,corpus,'matched_signed'):.3f} | {value(layer,corpus,'magnitude256'):.3f} |")
    lines += ["","A positive strict-core effect establishes genuinely recurring native gates. Equal-size magnitude is the important control: if it performs equally well, recurrence is descriptive rather than a uniquely privileged selection rule. The remainder of the 256-channel family bank is pair- or context-specific even when the strict core transfers."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained"); cells=[(0,"fineweb"),(0,"tinystories"),(1,"fineweb"),(1,"tinystories")]; x=np.arange(4); width=.2
    for offset,name in zip((-.3,-.1,.1,.3),("strict","matched_magnitude","matched_signed","magnitude256")):
        ax.bar(x+offset,[value(l,c,name) for l,c in cells],width,label=name)
    ax.set_xticks(x,["L0 Fine","L0 Tiny","L1 Fine","L1 Tiny"]); ax.set_ylabel("downstream fidelity"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.25); fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps(hypotheses,indent=2),flush=True); print(REPORT,flush=True)


if __name__ == "__main__": main()
