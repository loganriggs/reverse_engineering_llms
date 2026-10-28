#!/usr/bin/env python3
"""Test whether Layer-3 native product-channel circuits transfer when frozen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_native_channel_circuits import jaccard
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer3_frozen_channel_transfer.json")
OUTPUT = TENSOR_ROOT / "layer3_frozen_channel_transfer.json"
REPORT = TENSOR_ROOT / "LAYER3_FROZEN_CHANNEL_TRANSFER.md"
FIGURE = TENSOR_ROOT / "layer3_frozen_channel_transfer.png"


def recover_configs(config):
    native = json.loads(Path(config["source_config"]).read_text())
    null_config = json.loads(Path(native["source_config"]).read_text())
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
    return fixed, parent


@torch.inference_mode()
def channel_state(model, layer, base, counterfactual, final_position):
    block = model.blocks[layer]
    dimension = base["pre"].shape[-1]
    base_normalized = F.rms_norm(base["pre"], (dimension,))
    counterfactual_normalized = F.rms_norm(counterfactual["pre"], (dimension,))
    base_product = block.mlp.Left(base_normalized) * block.mlp.Right(base_normalized)
    counterfactual_product = (
        block.mlp.Left(counterfactual_normalized)
        * block.mlp.Right(counterfactual_normalized)
    )
    delta_product = counterfactual_product.float() - base_product.float()
    target_write = counterfactual["mlp_write"].float() - base["mlp_write"].float()
    target = target_write[0, : final_position + 1]
    denominator = target.square().sum().clamp_min(1e-30)
    scores = (
        delta_product[0, : final_position + 1]
        * (target @ block.mlp.Down.weight.float())
    ).sum(0) / denominator
    return {
        "base_product": base_product,
        "counterfactual_product": counterfactual_product,
        "delta_product": delta_product,
        "target_write": target_write,
        "scores": scores,
    }


def top_indices(score, rank):
    return score.abs().argsort(descending=True)[:rank]


@torch.inference_mode()
def evaluate_routes(model, layer, base, counterfactual, state, routes, final_position):
    block = model.blocks[layer]
    route_names = list(routes)
    products = []
    local = {}
    target_write = state["target_write"]
    for name in route_names:
        indices = routes[name]
        selected_delta = torch.zeros_like(state["delta_product"])
        selected_delta[..., indices] = state["delta_product"][..., indices]
        predicted = F.linear(selected_delta, block.mlp.Down.weight.float(), bias=None)
        local[name] = 1 - float(
            (predicted - target_write).square().sum()
            / target_write.square().sum().clamp_min(1e-30)
        )
        products.append(state["base_product"] + selected_delta.to(state["base_product"].dtype))
    products.append(state["counterfactual_product"])

    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    for product in products:
        outputs.append(base["pre"] + block.mlp.Down(product) + block.mlp.Down_bias)
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    semantic_target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, semantic_target, semantic_target)
    full_mlp_patch = logits[-1] - base_logits
    full_mlp_energy = fisher_inner(base_logits, full_mlp_patch, full_mlp_patch)
    result = {}
    for offset, name in enumerate(route_names):
        effect = logits[2 + offset] - base_logits
        fidelity, cosine, _ = fisher_fidelity(base_logits, full_mlp_patch, effect)
        result[name] = {
            "local_fidelity": local[name],
            "downstream_fidelity": fidelity,
            "downstream_cosine": cosine,
            "target_projection": fisher_inner(base_logits, effect, semantic_target)
            / max(target_energy, 1e-30),
            "energy_ratio": fisher_inner(base_logits, effect, effect)
            / max(full_mlp_energy, 1e-30),
        }
    result["full_mlp"] = {
        "target_projection": fisher_inner(base_logits, full_mlp_patch, semantic_target)
        / max(target_energy, 1e-30),
    }
    return result


def aggregate(records, route_names):
    return {
        name: {
            metric: median([record["routes"][name][metric] for record in records])
            for metric in (
                "local_fidelity", "downstream_fidelity", "downstream_cosine",
                "target_projection", "energy_ratio",
            )
        }
        for name in route_names
    } | {
        "full_mlp_target_projection": median([
            record["routes"]["full_mlp"]["target_projection"] for record in records
        ]),
        "support_jaccard": {
            name: median([record["support_jaccard"][name] for record in records])
            for name in route_names if name != "oracle"
        },
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    fixed, parent = recover_configs(config)
    saved = json.loads(Path(config["source_results"]).read_text())
    rank, layer = config["rank"], config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()

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
        target = int(record["pair"].split("->")[1])
        changed = raw.copy()
        changed[edit_position] = target
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
        return base[layer], counterfactual[layer]

    discovery_saved = [
        record for record in saved["semantic_records"]
        if record["corpus"] == config["discovery_corpus"]
        and record["context_index"] == config["discovery_context_index"]
    ]
    discovery = {}
    family_scores = {family: [] for family in parent["families"]}
    for record in discovery_saved:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        score = state["scores"].cpu()
        discovery[record["pair"]] = score
        family_scores[record["family"]].append(score.abs())
        print("discovery", record["family"], record["pair"], flush=True)

    pair_sets = {pair: top_indices(score, rank).cuda() for pair, score in discovery.items()}
    family_vectors = {
        family: torch.stack(scores).mean(0) for family, scores in family_scores.items()
    }
    family_sets = {
        family: top_indices(score, rank).cuda() for family, score in family_vectors.items()
    }
    global_set = top_indices(torch.stack(list(family_vectors.values())).mean(0), rank).cuda()
    family_names = list(parent["families"])
    wrong_family = {
        family: family_names[(index + 1) % len(family_names)]
        for index, family in enumerate(family_names)
    }
    generator = torch.Generator().manual_seed(config["seed"])
    channel_count = next(iter(discovery.values())).numel()
    random_set = torch.randperm(channel_count, generator=generator)[:rank].cuda()

    query_saved = [
        record for record in saved["semantic_records"]
        if record["context_index"] == config["evaluation_context_index"]
    ]
    route_names = ["oracle", "pair", "family", "global", "wrong_family", "random"]
    evaluated = []
    for record in query_saved:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, final_position)
        oracle = top_indices(state["scores"], rank)
        routes = {
            "oracle": oracle,
            "pair": pair_sets[record["pair"]],
            "family": family_sets[record["family"]],
            "global": global_set,
            "wrong_family": family_sets[wrong_family[record["family"]]],
            "random": random_set,
        }
        route_result = evaluate_routes(
            model, layer, base, counterfactual, state, routes, final_position,
        )
        evaluated.append({
            "corpus": record["corpus"],
            "family": record["family"],
            "pair": record["pair"],
            "row": record["row"],
            "source_start": record["source_start"],
            "routes": route_result,
            "support_jaccard": {
                name: jaccard(routes[name].tolist(), oracle.tolist())
                for name in route_names if name != "oracle"
            },
        })
        print("evaluation", record["corpus"], record["family"], record["pair"], flush=True)

    summaries = {
        corpus: aggregate(
            [record for record in evaluated if record["corpus"] == corpus], route_names,
        )
        for corpus in fixed["evaluation"]
    }
    fidelity = lambda corpus, route: summaries[corpus][route]["downstream_fidelity"]
    hypotheses = {
        "D1_oracle256_reproduces_mlp3_effect": {
            "supported": all(fidelity(c, "oracle") >= 0.50 for c in summaries),
        },
        "D2_pair_channels_transfer": {
            "supported": all(fidelity(c, "pair") >= 0.15 for c in summaries),
        },
        "D3_pair_beats_random": {
            "supported": all(fidelity(c, "pair") - fidelity(c, "random") >= 0.10 for c in summaries),
            "gaps": {c: fidelity(c, "pair") - fidelity(c, "random") for c in summaries},
        },
        "D4_family_beats_random": {
            "supported": all(fidelity(c, "family") - fidelity(c, "random") >= 0.05 for c in summaries),
            "gaps": {c: fidelity(c, "family") - fidelity(c, "random") for c in summaries},
        },
        "D5_family_is_semantically_specific": {
            "supported": all(fidelity(c, "family") - fidelity(c, "wrong_family") >= 0.05 for c in summaries),
            "gaps": {c: fidelity(c, "family") - fidelity(c, "wrong_family") for c in summaries},
        },
        "D6_family_beats_global": {
            "supported": all(fidelity(c, "family") - fidelity(c, "global") >= 0.03 for c in summaries),
            "gaps": {c: fidelity(c, "family") - fidelity(c, "global") for c in summaries},
        },
        "D7_family_transfers_ood": {
            "supported": fidelity("tinystories", "family") >= 0.75 * fidelity("fineweb", "family"),
            "retention": fidelity("tinystories", "family") / max(fidelity("fineweb", "family"), 1e-30),
        },
        "D8_frozen_support_is_nonrandom": {
            "supported": all(
                summaries[c]["support_jaccard"]["pair"] - summaries[c]["support_jaccard"]["random"] >= 0.03
                and summaries[c]["support_jaccard"]["family"] - summaries[c]["support_jaccard"]["random"] >= 0.03
                for c in summaries
            ),
        },
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "channel_count": channel_count,
        "wrong_family_mapping": wrong_family,
        "frozen_sets": {
            "pair": {key: value.cpu().tolist() for key, value in pair_sets.items()},
            "family": {key: value.cpu().tolist() for key, value in family_sets.items()},
            "global": global_set.cpu().tolist(),
            "random": random_set.cpu().tolist(),
        },
        "summaries": summaries,
        "records": evaluated,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Frozen native-channel transfer at Layer 3",
        "",
        "Native product channels are ranked only on FineWeb context 1. Pair-, family-, and global-level top-256 sets are then frozen and used to select the current query's product deltas on disjoint context 0 in FineWeb and TinyStories. Per-example oracle selection is a ceiling; cyclic wrong-family and fixed global/random sets are controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Frozen Layer-3 channels](layer3_frozen_channel_transfer.png)",
        "",
        "| Corpus | Route | Local fidelity | Downstream fidelity | Downstream cosine | Semantic target projection | Oracle support Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, summary in summaries.items():
        for name in route_names:
            overlap = 1.0 if name == "oracle" else summary["support_jaccard"][name]
            row = summary[name]
            lines.append(
                f"| {corpus} | {name} | {row['local_fidelity']:.3f} | "
                f"{row['downstream_fidelity']:.3f} | {row['downstream_cosine']:.3f} | "
                f"{row['target_projection']:.3f} | {overlap:.3f} |"
            )
    lines += [
        "",
        "The frozen sets patch the current example's own product changes; they never copy counterfactual activations from discovery examples. Thus success means a stable native address set, not merely activation similarity. Failure can mean that reusable computation lives in a rotated interaction subspace rather than in fixed native channels.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(route_names))
    for axis, corpus in zip(axes, ("fineweb", "tinystories")):
        axis.bar(x, [summaries[corpus][name]["downstream_fidelity"] for name in route_names])
        axis.set_xticks(x, route_names, rotation=35, ha="right")
        axis.set_ylabel("fidelity to full MLP3 patch")
        axis.set_title(corpus)
        axis.axhline(0, color="black", linewidth=.7)
        axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
