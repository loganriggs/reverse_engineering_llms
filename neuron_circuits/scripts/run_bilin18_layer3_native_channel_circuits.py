#!/usr/bin/env python3
"""Test native Layer-3 bilinear MLP channels as recurring semantic circuits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer4_residual_telescope_null import deranged_targets
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer3_native_channel_circuits.json")
OUTPUT = TENSOR_ROOT / "layer3_native_channel_circuits.json"
REPORT = TENSOR_ROOT / "LAYER3_NATIVE_CHANNEL_CIRCUITS.md"
FIGURE = TENSOR_ROOT / "layer3_native_channel_circuits.png"


def jaccard(first, second):
    first, second = set(first), set(second)
    return len(first & second) / max(len(first | second), 1)


@torch.inference_mode()
def analyze_channels(model, layer, base, counterfactual, ranks, final_position,
                     recurrence_rank):
    block = model.blocks[layer]
    dimension = base["pre"].shape[-1]
    base_normalized = F.rms_norm(base["pre"], (dimension,))
    counterfactual_normalized = F.rms_norm(counterfactual["pre"], (dimension,))
    base_product = block.mlp.Left(base_normalized) * block.mlp.Right(base_normalized)
    counterfactual_product = (
        block.mlp.Left(counterfactual_normalized) * block.mlp.Right(counterfactual_normalized)
    )
    delta_product = counterfactual_product.float() - base_product.float()
    target_write = counterfactual["mlp_write"].float() - base["mlp_write"].float()
    affected = slice(0, final_position + 1)
    target = target_write[0, affected]
    denominator = target.square().sum().clamp_min(1e-30)
    channel_scores = (
        delta_product[0, affected]
        * (target @ block.mlp.Down.weight.float())
    ).sum(0) / denominator
    order = channel_scores.abs().argsort(descending=True)

    local = {}
    patched_products = []
    for rank in ranks:
        indices = order[:rank]
        selected_delta = torch.zeros_like(delta_product)
        selected_delta[..., indices] = delta_product[..., indices]
        predicted = F.linear(
            selected_delta.float(), block.mlp.Down.weight.float(), bias=None,
        )
        fidelity = 1 - float(
            (predicted - target_write).square().sum() / target_write.square().sum().clamp_min(1e-30)
        )
        cosine = float(
            (predicted * target_write).sum()
            / (predicted.norm() * target_write.norm()).clamp_min(1e-30)
        )
        local[str(rank)] = {
            "fidelity": fidelity,
            "cosine": cosine,
            "signed_projection": float(channel_scores[indices].sum()),
        }
        patched_products.append(base_product + selected_delta.to(base_product.dtype))
    patched_products.append(counterfactual_product)

    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    for product in patched_products:
        write = block.mlp.Down(product) + block.mlp.Down_bias
        outputs.append(base["pre"] + write)
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    full_target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, full_target, full_target)
    full_mlp_index = 2 + len(ranks)
    full_mlp_patch = logits[full_mlp_index] - base_logits
    full_mlp_energy = fisher_inner(base_logits, full_mlp_patch, full_mlp_patch)
    downstream = {}
    for rank_index, rank in enumerate(ranks):
        effect = logits[2 + rank_index] - base_logits
        fidelity, cosine, _ = fisher_fidelity(base_logits, full_mlp_patch, effect)
        downstream[str(rank)] = {
            "fidelity_to_full_mlp_patch": fidelity,
            "cosine_to_full_mlp_patch": cosine,
            "target_projection": fisher_inner(base_logits, effect, full_target)
            / max(target_energy, 1e-30),
            "energy_ratio_to_full_mlp": fisher_inner(base_logits, effect, effect)
            / max(full_mlp_energy, 1e-30),
        }
    all_predicted = F.linear(
        delta_product.float(), block.mlp.Down.weight.float(), bias=None,
    )
    exact_error = float(
        (all_predicted - target_write).norm() / target_write.norm().clamp_min(1e-30)
    )
    return {
        "channel_count": int(delta_product.shape[-1]),
        "exact_recomposition_error": exact_error,
        "top_channels": [int(index) for index in order[:max(max(ranks), recurrence_rank)].tolist()],
        "top_channel_scores": [float(channel_scores[index]) for index in order[:max(max(ranks), recurrence_rank)]],
        "local": local,
        "full_mlp_patch": {
            "target_projection": fisher_inner(base_logits, full_mlp_patch, full_target)
            / max(target_energy, 1e-30),
            "fisher_energy": full_mlp_energy,
        },
        "downstream": downstream,
    }


def summarize_causal(records, ranks):
    return {
        "maximum_exact_recomposition_error": max(
            record["analysis"]["exact_recomposition_error"] for record in records
        ),
        "channel_count": records[0]["analysis"]["channel_count"],
        "local": {
            str(rank): {
                key: median([record["analysis"]["local"][str(rank)][key] for record in records])
                for key in ("fidelity", "cosine", "signed_projection")
            }
            for rank in ranks
        },
        "downstream": {
            str(rank): {
                key: median([record["analysis"]["downstream"][str(rank)][key] for record in records])
                for key in (
                    "fidelity_to_full_mlp_patch", "cosine_to_full_mlp_patch",
                    "target_projection", "energy_ratio_to_full_mlp",
                )
            }
            for rank in ranks
        },
        "full_mlp_target_projection": median([
            record["analysis"]["full_mlp_patch"]["target_projection"] for record in records
        ]),
    }


def recurrence_metrics(semantic_records, recurrence_rank):
    references = {
        record["pair"]: record
        for record in semantic_records
        if record["corpus"] == "fineweb" and record["context_index"] == 1
    }
    family_by_pair = {pair: record["family"] for pair, record in references.items()}
    results = {}
    for corpus in ("fineweb", "tinystories"):
        queries = [
            record for record in semantic_records
            if record["corpus"] == corpus and record["context_index"] == 0
        ]
        same_pair_jaccard, same_family, different_family = [], [], []
        for query in queries:
            query_set = query["analysis"]["top_channels"][:recurrence_rank]
            for pair, reference in references.items():
                score = jaccard(
                    query_set, reference["analysis"]["top_channels"][:recurrence_rank],
                )
                if pair == query["pair"]:
                    same_pair_jaccard.append(score)
                elif family_by_pair[pair] == query["family"]:
                    same_family.append(score)
                else:
                    different_family.append(score)
        results[corpus] = {
            "same_pair_jaccard": median(same_pair_jaccard),
            "same_family_jaccard": median(same_family),
            "different_family_jaccard": median(different_family),
            "family_gap": median(same_family) - median(different_family),
        }
    return results


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    null_config = json.loads(Path(config["source_config"]).read_text())
    telescope_config = json.loads(Path(null_config["source_config"]).read_text())
    embedding_config = json.loads(Path(telescope_config["source_config"]).read_text())
    recurrence_config = json.loads(Path(embedding_config["source_config"]).read_text())
    value_config = json.loads(Path(recurrence_config["source_config"]).read_text())
    factor_config = json.loads(Path(value_config["source_config"]).read_text())
    frozen_config = json.loads(Path(factor_config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    controls = deranged_targets(parent["families"], config["seed"])
    control_by_source = {row["source"]: row for row in controls}
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer = config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    semantic_records, null_records = [], []

    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    base_raw = np.asarray(
                        rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                    ).copy()
                    base_layers = forward_layers(
                        model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer,
                    )
                    semantic_raw = base_raw.copy()
                    semantic_raw[edit_position] = target_token
                    semantic_layers = forward_layers(
                        model, torch.as_tensor(semantic_raw, device="cuda")[None], last_layer=layer,
                    )
                    semantic_analysis = analyze_channels(
                        model, layer, base_layers[layer], semantic_layers[layer],
                        config["ranks"], final_position, config["recurrence_rank"],
                    )
                    semantic_records.append({
                        "corpus": corpus,
                        "context_index": context_index,
                        "family": family,
                        "pair": f"{source_token}->{target_token}",
                        "row": int(row_index),
                        "source_start": int(start),
                        "analysis": semantic_analysis,
                    })
                    if context_index == config["evaluation_context_index"]:
                        control = control_by_source[source_token]
                        null_raw = base_raw.copy()
                        null_raw[edit_position] = control["control_target"]
                        null_layers = forward_layers(
                            model, torch.as_tensor(null_raw, device="cuda")[None], last_layer=layer,
                        )
                        null_analysis = analyze_channels(
                            model, layer, base_layers[layer], null_layers[layer],
                            config["ranks"], final_position, config["recurrence_rank"],
                        )
                        null_records.append({
                            "corpus": corpus,
                            "context_index": context_index,
                            "family": family,
                            "pair": f"{source_token}->{control['control_target']}",
                            "semantic_pair": f"{source_token}->{target_token}",
                            "row": int(row_index),
                            "source_start": int(start),
                            "analysis": null_analysis,
                        })
                print(corpus, family, source_token, "->", target_token, flush=True)

    semantic_summary, null_summary = {}, {}
    same_context_jaccard = {}
    for corpus in fixed["evaluation"]:
        semantic_eval = [
            record for record in semantic_records
            if record["corpus"] == corpus and record["context_index"] == config["evaluation_context_index"]
        ]
        null_eval = [record for record in null_records if record["corpus"] == corpus]
        semantic_summary[corpus] = summarize_causal(semantic_eval, config["ranks"])
        null_summary[corpus] = summarize_causal(null_eval, config["ranks"])
        null_by_semantic = {record["semantic_pair"]: record for record in null_eval}
        same_context_jaccard[corpus] = median([
            jaccard(
                record["analysis"]["top_channels"][:config["recurrence_rank"]],
                null_by_semantic[record["pair"]]["analysis"]["top_channels"][:config["recurrence_rank"]],
            )
            for record in semantic_eval
        ])
    recurrence = recurrence_metrics(semantic_records, config["recurrence_rank"])
    maximum_error = max(
        max(summary[corpus]["maximum_exact_recomposition_error"] for corpus in summary)
        for summary in (semantic_summary, null_summary)
    )
    hypotheses = {
        "C1_native_channel_decomposition_is_exact": {
            "supported": maximum_error <= 1e-5,
            "maximum_relative_error": maximum_error,
        },
        "C2_top64_is_locally_sufficient": {
            "supported": all(semantic_summary[c]["local"]["64"]["fidelity"] >= 0.50 for c in semantic_summary),
        },
        "C3_top256_is_near_locally_sufficient": {
            "supported": all(semantic_summary[c]["local"]["256"]["fidelity"] >= 0.80 for c in semantic_summary),
        },
        "C4_top256_is_downstream_sufficient_for_mlp3": {
            "supported": all(
                semantic_summary[c]["downstream"]["256"]["fidelity_to_full_mlp_patch"] >= 0.50
                for c in semantic_summary
            ),
        },
        "C5_native_channels_recur_by_lexical_pair": {
            "supported": all(recurrence[c]["same_pair_jaccard"] >= 0.10 for c in recurrence),
        },
        "C6_native_channels_recur_by_semantic_family": {
            "supported": all(recurrence[c]["family_gap"] >= 0.02 for c in recurrence),
        },
        "C7_semantic_edits_are_more_channel_concentrated": {
            "supported": all(
                semantic_summary[c]["local"]["64"]["fidelity"]
                - null_summary[c]["local"]["64"]["fidelity"] >= 0.10
                for c in semantic_summary
            ),
        },
        "C8_semantic_and_null_channel_sets_differ": {
            "supported": all(value <= 0.25 for value in same_context_jaccard.values()),
            "same_context_jaccard": same_context_jaccard,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "semantic_summary": semantic_summary,
        "cross_family_null_summary": null_summary,
        "recurrence": recurrence,
        "same_context_semantic_null_jaccard": same_context_jaccard,
        "semantic_records": semantic_records,
        "null_records": null_records,
    }, indent=2))

    lines = [
        "# Native-channel circuits in the Layer-3 bilinear MLP",
        "",
        "For each semantic and cross-family-null token replacement, the exact Layer-3 product delta is decomposed into native channel writes `Down[:,c] Δh_c`. Channels are ranked by signed projection onto the complete MLP write delta. Nested channel sets are inserted into the base run, after which blocks 4–17 and the unembedding are recomputed. Channel-set recurrence is measured against disjoint FineWeb contexts.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-3 native channels](layer3_native_channel_circuits.png)",
        "",
        "| Corpus | Condition | Top64 local fidelity | Top256 local fidelity | Top256 downstream fidelity | Full MLP target projection |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus in semantic_summary:
        for condition, summary in (("semantic", semantic_summary[corpus]), ("cross-family null", null_summary[corpus])):
            lines.append(
                f"| {corpus} | {condition} | {summary['local']['64']['fidelity']:.3f} | "
                f"{summary['local']['256']['fidelity']:.3f} | "
                f"{summary['downstream']['256']['fidelity_to_full_mlp_patch']:.3f} | "
                f"{summary['full_mlp_target_projection']:.3f} |"
            )
    lines += [
        "",
        "| Query corpus | Same-pair Jaccard | Same-family / different-family | Family gap | Semantic/null same-context Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in recurrence.items():
        lines.append(
            f"| {corpus} | {row['same_pair_jaccard']:.3f} | "
            f"{row['same_family_jaccard']:.3f} / {row['different_family_jaccard']:.3f} | "
            f"{row['family_gap']:.3f} | {same_context_jaccard[corpus]:.3f} |"
        )
    lines += [
        "",
        f"Maximum all-channel recomposition error: `{maximum_error:.2e}`.",
        "",
        "A recurring channel set is evidence for stable native routing of a declared edit family, not automatically a monosemantic neuron. A channel can recur because it implements a reusable algebraic discrimination while its activation conditions remain heterogeneous. Conversely, low set recurrence can coexist with a stable non-native interaction subspace.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        ranks = config["ranks"]
        axis.plot(ranks, [semantic_summary[corpus]["local"][str(rank)]["fidelity"] for rank in ranks], marker="o", label="semantic/local")
        axis.plot(ranks, [null_summary[corpus]["local"][str(rank)]["fidelity"] for rank in ranks], marker="x", label="null/local")
        axis.plot(ranks, [semantic_summary[corpus]["downstream"][str(rank)]["fidelity_to_full_mlp_patch"] for rank in ranks], marker="s", label="semantic/downstream")
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, ranks, rotation=45)
        axis.set_title(corpus)
        axis.set_xlabel("native product channels")
        axis.set_ylabel("fidelity")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
