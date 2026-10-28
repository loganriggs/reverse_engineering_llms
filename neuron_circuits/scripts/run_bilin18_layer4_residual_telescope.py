#!/usr/bin/env python3
"""Exactly telescope the Layer-4 residual and trace it through the current-V relay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import edge_mask, median, output_with_attention
from run_bilin18_layer8_semantic_attention_factor_causality import vector_cosine
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer4_residual_telescope.json")
OUTPUT = TENSOR_ROOT / "layer4_residual_telescope.json"
REPORT = TENSOR_ROOT / "LAYER4_RESIDUAL_TELESCOPE.md"
FIGURE = TENSOR_ROOT / "layer4_residual_telescope.png"
COMPONENTS = (
    "initial_embedding",
    "repeat_embedding_l0", "repeat_embedding_l1", "repeat_embedding_l2",
    "repeat_embedding_l3", "repeat_embedding_l4",
    "attention_l0", "mlp_l0", "attention_l1", "mlp_l1",
    "attention_l2", "mlp_l2", "attention_l3", "mlp_l3",
)
EMBEDDING_COMPONENTS = COMPONENTS[:6]
ATTENTION_COMPONENTS = ("attention_l0", "attention_l1", "attention_l2", "attention_l3")
MLP_COMPONENTS = ("mlp_l0", "mlp_l1", "mlp_l2", "mlp_l3")


def product(values):
    result = 1.0
    for value in values:
        result = result * value
    return result


@torch.inference_mode()
def telescope_mixed_components(model, base_layers, counterfactual_layers, target_layer):
    dx0 = counterfactual_layers[0]["x0"].float() - base_layers[0]["x0"].float()
    coefficients = [model.blocks[layer].lambdas[0].float() for layer in range(target_layer + 1)]
    injections = [model.blocks[layer].lambdas[1].float() for layer in range(target_layer + 1)]
    target_residual_scale = coefficients[target_layer]
    components = {
        "initial_embedding": (
            target_residual_scale * product(coefficients[:target_layer]) * dx0
        ),
    }
    for layer in range(target_layer):
        future = product(coefficients[layer + 1:target_layer])
        propagation = target_residual_scale * future
        components[f"repeat_embedding_l{layer}"] = propagation * injections[layer] * dx0
        components[f"attention_l{layer}"] = propagation * (
            counterfactual_layers[layer]["attention_write"].float()
            - base_layers[layer]["attention_write"].float()
        )
        components[f"mlp_l{layer}"] = propagation * (
            counterfactual_layers[layer]["mlp_write"].float()
            - base_layers[layer]["mlp_write"].float()
        )
    components[f"repeat_embedding_l{target_layer}"] = injections[target_layer] * dx0
    target = (
        counterfactual_layers[target_layer]["mixed"].float()
        - base_layers[target_layer]["mixed"].float()
    )
    reconstructed = sum(components.values())
    error = float((reconstructed - target).norm() / target.norm().clamp_min(1e-30))
    return components, target, error


def rmsnorm_jvp(state, directions):
    """Exact JVP of unweighted RMSNorm for a batch of directions."""
    epsilon = torch.finfo(state.dtype).eps
    radius = (state.square().mean(-1, keepdim=True) + epsilon).sqrt()
    dot = (directions * state).mean(-1, keepdim=True)
    return directions / radius - state * dot / radius.pow(3)


@torch.inference_mode()
def integrated_component_writes(model, layer, base, counterfactual, components, mask,
                                quadrature_points):
    block = model.blocks[layer]
    names = list(COMPONENTS)
    directions = torch.cat([components[name] for name in names], dim=0)
    base_mixed = base["mixed"].float()
    counterfactual_mixed = counterfactual["mixed"].float()
    delta_mixed = counterfactual_mixed - base_mixed
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    current_contributions = torch.zeros_like(directions)
    route_contributions = torch.zeros_like(directions)
    pattern = (
        base["details"]["score1"].float()
        * base["details"]["score2"].float()
        * mask
    )[0]
    for node, weight in zip(nodes, weights):
        state = base_mixed + float(node) * delta_mixed
        normalized_jvp = rmsnorm_jvp(state, directions)
        current_jvp = F.linear(
            normalized_jvp, block.attn.c_v.weight.float(), bias=None,
        )
        current_contributions += float(weight) * current_jvp
        count, length, dimension = current_jvp.shape
        heads, head_dimension = block.attn.n_head, block.attn.head_dim
        current_heads = current_jvp.view(count, length, heads, head_dimension)
        mixed_jvp = (1 - block.attn.lamb.float()) * current_heads
        attended = torch.einsum("hqk,ckhd->chqd", pattern, mixed_jvp)
        flattened = attended.transpose(1, 2).contiguous().view(count, length, dimension)
        route_contributions += float(weight) * F.linear(
            flattened, block.attn.c_proj.weight.float(), bias=None,
        )
    base_normalized = F.rms_norm(base_mixed, (base_mixed.shape[-1],))
    counterfactual_normalized = F.rms_norm(
        counterfactual_mixed, (counterfactual_mixed.shape[-1],),
    )
    current_target = (
        block.attn.c_v(counterfactual_normalized)
        - block.attn.c_v(base_normalized)
    )
    batch, length, dimension = current_target.shape
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    current_target_heads = current_target.view(batch, length, heads, head_dimension)
    attended_target = torch.einsum(
        "hqk,bkhd->bhqd", pattern,
        (1 - block.attn.lamb.float()) * current_target_heads,
    )
    route_target = F.linear(
        attended_target.transpose(1, 2).contiguous().view(batch, length, dimension),
        block.attn.c_proj.weight.float(), bias=None,
    )
    current_error = float(
        (current_contributions.sum(0, keepdim=True) - current_target).norm()
        / current_target.norm().clamp_min(1e-30)
    )
    route_error = float(
        (route_contributions.sum(0, keepdim=True) - route_target).norm()
        / route_target.norm().clamp_min(1e-30)
    )
    return {
        name: route_contributions[index:index + 1]
        for index, name in enumerate(names)
    }, current_error, route_error


@torch.inference_mode()
def analyze_example(model, layer, base_layers, counterfactual_layers, edges,
                    final_position, quadrature_points):
    base, counterfactual = base_layers[layer], counterfactual_layers[layer]
    components, mixed_target, telescope_error = telescope_mixed_components(
        model, base_layers, counterfactual_layers, layer,
    )
    mask = edge_mask(
        model.config.n_head, base["output"].shape[1], edges, base["output"].device,
    )
    writes, current_error, route_error = integrated_component_writes(
        model, layer, base, counterfactual, components, mask, quadrature_points,
    )
    route_target = sum(writes.values())
    local_energy = float(route_target.square().sum().clamp_min(1e-30))
    local_projection = {
        name: float((write * route_target).sum() / local_energy)
        for name, write in writes.items()
    }
    absolute_total = sum(abs(value) for value in local_projection.values())
    top4_share = sum(sorted(
        (abs(value) for value in local_projection.values()), reverse=True,
    )[:4]) / max(absolute_total, 1e-30)

    groups = {
        **{name: writes[name] for name in COMPONENTS},
        "embedding_all": sum(writes[name] for name in EMBEDDING_COMPONENTS),
        "attention_all": sum(writes[name] for name in ATTENTION_COMPONENTS),
        "mlp_all": sum(writes[name] for name in MLP_COMPONENTS),
        "learned_writes": sum(writes[name] for name in ATTENTION_COMPONENTS + MLP_COMPONENTS),
        "early_writes": sum(writes[name] for name in ("attention_l0", "mlp_l0", "attention_l1", "mlp_l1")),
        "later_writes": sum(writes[name] for name in ("attention_l2", "mlp_l2", "attention_l3", "mlp_l3")),
        "all_components": route_target,
    }
    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    group_indices = {}
    for name, write in groups.items():
        group_indices[name] = len(outputs)
        outputs.append(output_with_attention(
            model.blocks[layer], base, base["attention_write"] + write,
        ))
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, target, target)
    downstream_projection = {
        name: fisher_inner(base_logits, logits[index] - base_logits, target)
        / max(target_energy, 1e-30)
        for name, index in group_indices.items()
    }
    raw_projection = {
        name: float((component * mixed_target).sum() / mixed_target.square().sum().clamp_min(1e-30))
        for name, component in components.items()
    }
    return {
        "telescope_error": telescope_error,
        "current_value_integration_error": current_error,
        "route_integration_error": route_error,
        "local_projection": local_projection,
        "raw_mixed_projection": raw_projection,
        "top4_absolute_local_share": top4_share,
        "downstream_projection": downstream_projection,
    }


def summarize(records):
    return {
        "maximum_telescope_error": max(row["telescope_error"] for row in records),
        "maximum_current_value_integration_error": max(
            row["current_value_integration_error"] for row in records
        ),
        "maximum_route_integration_error": max(row["route_integration_error"] for row in records),
        "top4_absolute_local_share": median([
            row["top4_absolute_local_share"] for row in records
        ]),
        "components": {
            name: {
                "local_projection": median([row["local_projection"][name] for row in records]),
                "raw_mixed_projection": median([row["raw_mixed_projection"][name] for row in records]),
                "downstream_projection": median([row["downstream_projection"][name] for row in records]),
            }
            for name in COMPONENTS
        },
        "groups": {
            name: median([row["downstream_projection"][name] for row in records])
            for name in (
                "embedding_all", "attention_all", "mlp_all", "learned_writes",
                "early_writes", "later_writes", "all_components",
            )
        },
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    embedding_config = json.loads(Path(config["source_config"]).read_text())
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
    frozen = json.loads(Path(config["frozen_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer = config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    edges = [
        tuple(row["edge"])
        for row in frozen["discovered_routes"][str(layer)]["magnitude"][:config["route_rank"]]
    ]
    results, records_by_corpus = {}, {}
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                row_index, start = windows[config["evaluation_context_index"]]
                base_raw = np.asarray(
                    rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                ).copy()
                counterfactual_raw = base_raw.copy()
                counterfactual_raw[edit_position] = target_token
                base_layers = forward_layers(
                    model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer,
                )
                counterfactual_layers = forward_layers(
                    model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=layer,
                )
                record = analyze_example(
                    model, layer, base_layers, counterfactual_layers, edges,
                    final_position, config["quadrature_points"],
                )
                record.update({
                    "family": family,
                    "pair": f"{source_token}->{target_token}",
                    "row": int(row_index),
                    "source_start": int(start),
                })
                records.append(record)
                print(corpus, family, source_token, "->", target_token, flush=True)
        records_by_corpus[corpus] = records
        results[corpus] = summarize(records)

    profiles = {
        corpus: [abs(results[corpus]["components"][name]["local_projection"]) for name in COMPONENTS]
        for corpus in results
    }
    profile_cosine = vector_cosine(profiles["fineweb"], profiles["tinystories"])
    maximum_telescope = max(result["maximum_telescope_error"] for result in results.values())
    maximum_current = max(
        result["maximum_current_value_integration_error"] for result in results.values()
    )
    maximum_route = max(result["maximum_route_integration_error"] for result in results.values())
    hypotheses = {
        "Z1_residual_telescope_is_exact": {
            "supported": maximum_telescope <= 1e-5,
            "maximum_relative_error": maximum_telescope,
        },
        "Z2_integrated_current_value_recomposes": {
            "supported": maximum_current <= 1e-4,
            "maximum_relative_error": maximum_current,
        },
        "Z3_integrated_route_recomposes": {
            "supported": maximum_route <= 1e-4,
            "maximum_relative_error": maximum_route,
        },
        "Z4_learned_writes_beat_embedding_paths": {
            "supported": all(
                abs(results[corpus]["groups"]["learned_writes"])
                > abs(results[corpus]["groups"]["embedding_all"])
                for corpus in results
            ),
        },
        "Z5_four_components_dominate_locally": {
            "supported": all(results[corpus]["top4_absolute_local_share"] >= 0.75 for corpus in results),
        },
        "Z6_component_profile_transfers_ood": {
            "supported": profile_cosine >= 0.80,
            "cosine": profile_cosine,
        },
        "Z7_later_blocks_dominate_earlier_blocks": {
            "supported": all(
                abs(results[corpus]["groups"]["later_writes"])
                > abs(results[corpus]["groups"]["early_writes"])
                for corpus in results
            ),
        },
        "Z8_current_value_route_is_downstream_nontrivial": {
            "supported": all(
                results[corpus]["groups"]["all_components"] >= 0.05 for corpus in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "component_profile_cosine": profile_cosine,
        "results": results,
        "records": records_by_corpus,
    }, indent=2))

    lines = [
        "# Exact upstream telescope of the Layer-4 current-value relay",
        "",
        "The Layer-4 mixed-state edit is decomposed exactly using `x_(l+1)=a_l x_l+b_l x0+attention_l+MLP_l`. Fourteen additive terms distinguish the initial embedding, repeated embedding injections at layers 0–4, and attention/MLP writes at layers 0–3. A 16-point Gauss–Legendre integral of the analytic RMSNorm Jacobian maps each term through Layer-4 current-V and the frozen attention skeleton. Resulting write components are patched through the Layer-4 MLP and complete suffix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-4 residual telescope](layer4_residual_telescope.png)",
        "",
        "| Corpus | Component | Raw mixed projection | Local routed projection | Individual downstream projection |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for name in COMPONENTS:
            row = result["components"][name]
            lines.append(
                f"| {corpus} | {name} | {row['raw_mixed_projection']:.3f} | "
                f"{row['local_projection']:.3f} | {row['downstream_projection']:.3f} |"
            )
    lines += [
        "",
        "| Corpus | Embedding group | Attention group | MLP group | Learned writes | Early / later writes | All current-V route | Top-4 local share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        groups = result["groups"]
        lines.append(
            f"| {corpus} | {groups['embedding_all']:.3f} | {groups['attention_all']:.3f} | "
            f"{groups['mlp_all']:.3f} | {groups['learned_writes']:.3f} | "
            f"{groups['early_writes']:.3f} / {groups['later_writes']:.3f} | "
            f"{groups['all_components']:.3f} | {result['top4_absolute_local_share']:.3f} |"
        )
    lines += [
        "",
        f"Maximum residual telescope error: `{maximum_telescope:.2e}`; current-V integral error: `{maximum_current:.2e}`; routed-write integral error: `{maximum_route:.2e}`. Cross-corpus absolute component-profile cosine: `{profile_cosine:.3f}`.",
        "",
        "The telescope is an exact algebraic decomposition of a particular state difference. Integrated-Jacobian terms allocate its nonlinear RMSNorm/current-V transformation along the straight base-to-counterfactual path. Individual finite patches need not add after the Layer-4 MLP and later nonlinear blocks; group patches are reported separately for that reason.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(COMPONENTS))
    width = 0.36
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "tinystories")):
        axes[0].bar(
            x + offset,
            [results[corpus]["components"][name]["local_projection"] for name in COMPONENTS],
            width,
            label=corpus,
        )
        axes[1].bar(
            x + offset,
            [results[corpus]["components"][name]["downstream_projection"] for name in COMPONENTS],
            width,
            label=corpus,
        )
    labels = [name.replace("repeat_embedding_", "emb_").replace("attention_", "attn_") for name in COMPONENTS]
    for axis, title in zip(axes, ("Integrated local route", "Finite downstream patch")):
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
