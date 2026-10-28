#!/usr/bin/env python3
"""Measure per-family causal input-mode K90 at tangent and full edit scales."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_input_form_k90.json")
OUTPUT = TENSOR_ROOT / "layer3_input_form_k90.json"
REPORT = TENSOR_ROOT / "LAYER3_INPUT_FORM_K90.md"
FIGURE = TENSOR_ROOT / "layer3_input_form_k90.png"


def first_rank(curve, ranks, threshold):
    for rank in ranks:
        if curve[str(rank)] >= threshold:
            return rank
    return None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    null_config = json.loads(Path(config["source_config"]).read_text())
    input_config = json.loads(Path(null_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(form_config["native_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    layer, output_rank = form_config["layer"], input_config["output_rank"]
    ranks, scales = config["ranks"], config["scales"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[layer]
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    specs = fixed["evaluation"]

    def reconstruct(record):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(
            rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
        return base[layer], counterfactual[layer]

    discovery_records = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    discovery_scores, family_by_pair = {}, {}
    for record in discovery_records:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        discovery_scores[record["pair"]] = state["scores"].abs().cpu()
        family_by_pair[record["pair"]] = record["family"]
    pair_cache = {}
    for pair, family in family_by_pair.items():
        indices = top_indices(torch.stack([
            score for candidate_pair, score in discovery_scores.items()
            if family_by_pair[candidate_pair] == family and candidate_pair != pair
        ]).mean(0), frozen_config["rank"]).cuda()
        metric = input_metric(block, indices, output_rank)
        pair_cache[pair] = {
            "indices": indices,
            "output": metric["output"],
            "vectors": metric["vectors"],
        }

    query_records = [
        record for record in native_saved["semantic_records"]
        if record["context_index"] == frozen_config["evaluation_context_index"]
    ]
    evaluated = []
    all_indices = torch.arange(block.mlp.Left.weight.shape[0], device="cuda")
    for record in query_records:
        base, counterfactual = reconstruct(record)
        cache = pair_cache[record["pair"]]
        dimension = base["pre"].shape[-1]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        delta = xc - xb
        scale_results = {}
        for scale in scales:
            changed_full = xb + scale * delta
            full_mlp = bank_delta(block, xb, changed_full, all_indices)
            exact_bank = bank_delta(block, xb, changed_full, cache["indices"])
            core_exact = (exact_bank @ cache["output"]) @ cache["output"].T
            writes = {"full_mlp": full_mlp, "family_exact": core_exact}
            for rank in ranks:
                chosen = cache["vectors"][:, :rank]
                changed = xb + scale * (delta @ chosen) @ chosen.T
                predicted = bank_delta(block, xb, changed, cache["indices"])
                writes[f"rank_{rank}"] = (predicted @ cache["output"]) @ cache["output"].T
            effects = evaluate_writes(model, layer, base, counterfactual, writes, final_position)
            scale_results[str(scale)] = {
                str(rank): effects[f"rank_{rank}"]["family_patch_fidelity"] for rank in ranks
            }
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "scales": scale_results,
        })
        print("curve", record["corpus"], record["family"], record["pair"], flush=True)

    by_family = {}
    for corpus in fixed["evaluation"]:
        corpus_records = [record for record in evaluated if record["corpus"] == corpus]
        by_family[corpus] = {}
        for family in parent["families"]:
            records = [record for record in corpus_records if record["family"] == family]
            by_family[corpus][family] = {}
            for scale in scales:
                curve = {
                    str(rank): median([record["scales"][str(scale)][str(rank)] for record in records])
                    for rank in ranks
                }
                by_family[corpus][family][str(scale)] = {
                    "curve": curve,
                    "causal_k90": first_rank(curve, ranks, config["threshold"]),
                }

    full_k90 = {
        corpus: {family: row["1.0"]["causal_k90"] for family, row in families.items()}
        for corpus, families in by_family.items()
    }
    tangent_k90 = {
        corpus: {family: row["0.1"]["causal_k90"] for family, row in families.items()}
        for corpus, families in by_family.items()
    }
    finite = lambda value: value if value is not None else 10**9
    hypotheses = {
        "K1_gender_reaches_k90_by64": {
            "supported": all(finite(full_k90[c]["gender"]) <= 64 for c in full_k90),
            "values": {c: full_k90[c]["gender"] for c in full_k90},
        },
        "K2_increasing_scale_reaches_k90_by128": {
            "supported": all(finite(full_k90[c]["increasing_scale"]) <= 128 for c in full_k90),
            "values": {c: full_k90[c]["increasing_scale"] for c in full_k90},
        },
        "K3_every_family_reaches_k90_by128": {
            "supported": all(finite(value) <= 128 for families in full_k90.values() for value in families.values()),
        },
        "K4_full_scale_k90_is_ood_stable": {
            "supported": all(
                max(finite(full_k90["fineweb"][family]), finite(full_k90["tinystories"][family]))
                <= 2 * min(finite(full_k90["fineweb"][family]), finite(full_k90["tinystories"][family]))
                for family in parent["families"]
            ),
        },
        "K5_tangent_rank_is_comparable": {
            "supported": all(
                finite(tangent_k90[corpus][family]) <= 2 * finite(full_k90[corpus][family])
                for corpus in full_k90 for family in parent["families"]
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "by_family": by_family,
        "records": evaluated,
    }, indent=2))

    lines = [
        "# Per-family causal K90 of Layer-3 input forms",
        "",
        "The correct family-bank input eigenspace is swept through rank 512. `Causal K90` is the first tested rank whose median held-out downstream fidelity to the exact four-axis output-core patch reaches 0.90. It is reported separately by semantic family, corpus, and 10% versus full edit scale.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Per-family input K90](layer3_input_form_k90.png)",
        "",
        "| Corpus | Family | K90 at 10% | K90 at 100% | Fidelity curve at 100% (ranks 4→512) |",
        "|---|---|---:|---:|---|",
    ]
    for corpus, families in by_family.items():
        for family, row in families.items():
            tangent = row["0.1"]["causal_k90"]
            full = row["1.0"]["causal_k90"]
            curve = ", ".join(f"{row['1.0']['curve'][str(rank)]:.3f}" for rank in ranks)
            lines.append(
                f"| {corpus} | {family} | {tangent if tangent is not None else '>512'} | "
                f"{full if full is not None else '>512'} | {curve} |"
            )
    lines += [
        "",
        "A low K90 is a use-case-specific notion of mechanistic simplicity: how many weight-derived input directions are sufficient for this family’s clean output core under a declared intervention. It is not a neuron count and does not measure the rank needed for the entire MLP write or language-model behavior.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(parent["families"])))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        for color, family in zip(colors, parent["families"]):
            curve = by_family[corpus][family]["1.0"]["curve"]
            axis.plot(ranks, [curve[str(rank)] for rank in ranks], marker="o", label=family, color=color)
        axis.axhline(config["threshold"], color="black", linestyle="--", linewidth=.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, ranks)
        axis.set_xlabel("input-mode rank")
        axis.set_ylabel("core-patch fidelity")
        axis.set_title(corpus)
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
