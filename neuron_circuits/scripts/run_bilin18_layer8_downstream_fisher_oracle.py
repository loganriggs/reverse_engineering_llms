#!/usr/bin/env python3
"""Compare local tensor-score and downstream-Fisher definitions of important modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    pearson,
    prefix_to_layer,
    repeat_batch,
    route_indices,
    select_sites,
    suffix_from_layer,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_downstream_fisher_oracle.json")
OUTPUT = TENSOR_ROOT / "layer8_downstream_fisher_oracle.json"
REPORT = TENSOR_ROOT / "LAYER8_DOWNSTREAM_FISHER_ORACLE.md"
FIGURE = TENSOR_ROOT / "layer8_downstream_fisher_oracle.png"
ROUTES = ("local_oracle", "fisher_oracle", "flat256", "hierarchical128", "fixed16")


def centered_fisher_impact(gradients, mode_write_tangents):
    centered = gradients - gradients.mean(0, keepdim=True)
    response = centered @ mode_write_tangents.T
    return response.square().sum(0) / max(len(gradients) - 1, 1)


def matched_local_controls(local_magnitude, fisher_impact, selected, count):
    """Match selected modes to controls with similar local contribution."""
    selected = selected[:count].tolist()
    excluded = set(torch.topk(fisher_impact, min(32, len(fisher_impact))).indices.tolist())
    available = [index for index in range(len(local_magnitude)) if index not in excluded]
    controls = []
    for index in selected:
        target = float(torch.log10(local_magnitude[index].clamp_min(1e-30)))
        control = min(
            (candidate for candidate in available if candidate not in controls),
            key=lambda candidate: abs(float(torch.log10(local_magnitude[candidate].clamp_min(1e-30))) - target),
        )
        controls.append(control)
    return selected, controls


def mode_write_tangents(block, x, mode_directions):
    """Exact bilinear write tangents for literal removal of each mode component."""
    left_x = block.mlp.Left(x).float()
    right_x = block.mlp.Right(x).float()
    left_u = F.linear(mode_directions.T, block.mlp.Left.weight.float())
    right_u = F.linear(mode_directions.T, block.mlp.Right.weight.float())
    product_tangent = left_u * right_x[None] + right_u * left_x[None]
    write_tangent = F.linear(product_tangent, block.mlp.Down.weight.float())
    amplitudes = -(x @ mode_directions)
    return write_tangent * amplitudes[:, None], amplitudes


def fisher_gradient_sketch(model, base_state, x0, value_bus, position, samples, seed):
    """Monte Carlo score-function sketch of the suffix Fisher metric."""
    dimension = base_state.shape[-1]
    delta = torch.zeros(dimension, device=base_state.device, requires_grad=True)
    mask = torch.zeros_like(base_state)
    mask[0, position] = delta
    x = base_state + mask
    bus = value_bus
    for block in model.blocks[9:]:
        x, bus = block(x, bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    logits = 30 * torch.tanh(raw / 30)
    logp = F.log_softmax(logits, -1)
    probabilities = logp.exp().detach()
    generator = torch.Generator(device=base_state.device).manual_seed(seed)
    tokens = torch.multinomial(
        probabilities[0], 2 * samples, replacement=True, generator=generator,
    )
    gradients = []
    for index, token in enumerate(tokens):
        gradients.append(torch.autograd.grad(
            logp[0, token], delta, retain_graph=index + 1 < len(tokens),
        )[0].detach())
    gradients = torch.stack(gradients)
    return gradients[:samples], gradients[samples:], logits.detach()[0]


@torch.inference_mode()
def finite_routes(model, pre, normalized, x0, value_bus, position, basis, vectors,
                  query, indices, direction, edit_norm_fraction):
    block = model.blocks[8]
    variants = normalized.repeat(len(ROUTES) + 1, 1, 1)
    x = normalized[0, position].float()
    target_norm = edit_norm_fraction * x.norm()
    for route_index, route in enumerate(ROUTES, start=1):
        mode_directions = basis @ vectors[query, :, indices[route]]
        raw_delta = -(x @ mode_directions) @ mode_directions.T
        delta = raw_delta * (target_norm / raw_delta.norm().clamp_min(1e-20))
        variants[route_index, position] = (x + delta).to(variants.dtype)
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(ROUTES) + 1) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(ROUTES) + 1),
        repeat_batch(value_bus, len(ROUTES) + 1), outputs, position,
    )
    native_logits = logits[0]
    native_write = writes[0, position].float()
    result = {}
    for route_index, route in enumerate(ROUTES, start=1):
        result[route] = {
            "logit_effect": logits[route_index] - native_logits,
            "native_kl": exact_kl(native_logits, logits[route_index]),
            "direct_query_effect": float(
                (writes[route_index, position].float() - native_write) @ direction
            ),
        }
    return native_logits, result


@torch.inference_mode()
def finite_single_modes(model, pre, normalized, x0, value_bus, position,
                        mode_directions, amplitudes, mode_indices, removal_fraction):
    block = model.blocks[8]
    variants = normalized.repeat(len(mode_indices) + 1, 1, 1)
    x = normalized[0, position].float()
    for variant, mode in enumerate(mode_indices, start=1):
        delta = removal_fraction * amplitudes[mode] * mode_directions[:, mode]
        variants[variant, position] = (x + delta).to(variants.dtype)
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(mode_indices) + 1) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(mode_indices) + 1),
        repeat_batch(value_bus, len(mode_indices) + 1), outputs, position,
    )
    return [exact_kl(logits[0], logits[index]) for index in range(1, len(logits))]


def summarize(records):
    correlations = np.asarray([r["sketch_impact_correlation"] for r in records], dtype=np.float64)
    recalls = np.asarray([r["sketch_top16_recall"] for r in records], dtype=np.float64)
    sketch = {
        "valid_fraction": float(np.isfinite(correlations).mean()),
        "median_impact_correlation": float(np.nanmedian(correlations)),
        "median_top16_recall": float(np.nanmedian(recalls)),
    }
    routes = {}
    for route in ROUTES:
        routes[route] = {
            "median_native_kl": float(np.median([r["routes"][route]["native_kl"] for r in records])),
            "median_absolute_direct_query_effect": float(np.median([
                abs(r["routes"][route]["direct_query_effect"]) for r in records
            ])),
            "median_local_support_recall": float(np.median([
                r["routes"][route]["local_support_recall"] for r in records
            ])),
            "median_fisher_support_recall": float(np.median([
                r["routes"][route]["fisher_support_recall"] for r in records
            ])),
            "median_fidelity_to_local_effect": float(np.median([
                r["routes"][route]["fidelity_to_local_effect"] for r in records
            ])),
            "median_fidelity_to_fisher_effect": float(np.median([
                r["routes"][route]["fidelity_to_fisher_effect"] for r in records
            ])),
        }
    pair_ratios = []
    wins = []
    for record in records:
        for top, control in zip(record["single_mode_top_kl"], record["single_mode_control_kl"]):
            pair_ratios.append(top / max(control, 1e-30))
            wins.append(top > control)
    finite = {
        "median_paired_kl_ratio": float(np.median(pair_ratios)),
        "paired_win_rate": float(np.mean(wins)),
    }
    return {"sketch": sketch, "routes": routes, "finite_single_modes": finite}


def evaluate_corpus(model, config, corpus_config, basis, directions, values, vectors, corpus_index):
    scan_config = {**config, "positions": [config["causal_position"]]}
    states, _ = collect(model, 8, corpus_config, scan_config)
    selected_sites = select_sites(states, basis, values, vectors, config["sites_per_query"])
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    block = model.blocks[8]
    position = config["causal_position"]
    records = []
    for query, sites in enumerate(selected_sites):
        mode_directions = basis @ vectors[query]
        for site_number, row_index in enumerate(sites):
            ids = torch.as_tensor(
                np.asarray(rows[row_index, :position + 1], dtype=np.int64), device="cuda",
            )[None]
            with torch.no_grad():
                x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
                native_write = block.mlp.Down(
                    block.mlp.Left(normalized) * block.mlp.Right(normalized)
                ) + block.mlp.Down_bias
                base_state = (pre + native_write).detach()
                x = normalized[0, position].float()
                z = x @ basis
                routed = route_indices(
                    z, values, vectors, config["budget"], config["coarse_dimension"],
                    config["refine_dimension"], config["shortlist"],
                )
                mode_tangents, amplitudes = mode_write_tangents(
                    block, x, mode_directions,
                )
            seed = config["seed"] + corpus_index * 100000 + query * 100 + site_number
            first_gradients, second_gradients, _ = fisher_gradient_sketch(
                model, base_state, x0.detach(), value_bus.detach(), position,
                config["fisher_samples_per_split"], seed,
            )
            first_impact = centered_fisher_impact(first_gradients, mode_tangents)
            second_impact = centered_fisher_impact(second_gradients, mode_tangents)
            fisher_indices = first_impact.topk(config["budget"]).indices
            second_indices = second_impact.topk(config["budget"]).indices
            local_contribution = (values[query] * (x @ mode_directions).square()).abs()
            local_indices = local_contribution.topk(config["budget"]).indices
            indices = {
                "local_oracle": local_indices,
                "fisher_oracle": fisher_indices,
                "flat256": routed["flat256"][query],
                "hierarchical128": routed["hierarchical128"][query],
                "fixed16": routed["fixed16"][query],
            }
            with torch.no_grad():
                native_logits, route_rows = finite_routes(
                    model, pre, normalized, x0, value_bus, position, basis, vectors,
                    query, indices, directions[:, query], config["edit_norm_fraction"],
                )
                chosen_top, matched_controls = matched_local_controls(
                    local_contribution, first_impact, fisher_indices,
                    config["finite_modes_per_site"],
                )
                single_kl = finite_single_modes(
                    model, pre, normalized, x0, value_bus, position,
                    mode_directions, amplitudes, chosen_top + matched_controls,
                    config["component_removal_fraction"],
                )
            local_set = set(local_indices.cpu().tolist())
            fisher_set = set(fisher_indices.cpu().tolist())
            local_effect = route_rows["local_oracle"]["logit_effect"]
            fisher_effect = route_rows["fisher_oracle"]["logit_effect"]
            for route in ROUTES:
                route_set = set(indices[route].cpu().tolist())
                local_fidelity, _, _ = fisher_fidelity(
                    native_logits, local_effect, route_rows[route]["logit_effect"],
                )
                fisher_match, _, _ = fisher_fidelity(
                    native_logits, fisher_effect, route_rows[route]["logit_effect"],
                )
                route_rows[route].pop("logit_effect")
                route_rows[route].update({
                    "local_support_recall": len(route_set & local_set) / config["budget"],
                    "fisher_support_recall": len(route_set & fisher_set) / config["budget"],
                    "fidelity_to_local_effect": local_fidelity,
                    "fidelity_to_fisher_effect": fisher_match,
                    "indices": sorted(route_set),
                })
            impact_correlation = pearson(
                first_impact.cpu().numpy(), second_impact.cpu().numpy(),
            )
            sketch_valid = bool(np.isfinite(impact_correlation))
            records.append({
                "query": query,
                "row": int(row_index),
                "sketch_valid": sketch_valid,
                "sketch_impact_correlation": impact_correlation,
                "sketch_top16_recall": (
                    len(set(fisher_indices.cpu().tolist()) & set(second_indices.cpu().tolist()))
                    / config["budget"] if sketch_valid else float("nan")
                ),
                "routes": route_rows,
                "single_mode_top_kl": single_kl[:config["finite_modes_per_site"]],
                "single_mode_control_kl": single_kl[config["finite_modes_per_site"]:],
            })
            del first_gradients, second_gradients
        print("query", query, "complete", flush=True)
    return {"summary": summarize(records), "records": records}


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    discovery_config = {**config, "positions": config["discovery_positions"]}
    with torch.no_grad():
        discovery_x, discovery_write = collect(
            model, 8, config["discovery"], discovery_config,
        )
        _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        basis = state_vh.T[:, :config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        directions = write_vh.T[:, :config["queries"]]
        block = model.blocks[8]
        left = block.mlp.Left.weight.float() @ basis
        right = block.mlp.Right.weight.float() @ basis
        coefficients = block.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        print("downstream Fisher", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, directions, values, vectors, corpus_index,
        )

    hypotheses = {
        "F1_fisher_sketch_is_stable": {
            "supported": all(
                result["summary"]["sketch"]["median_impact_correlation"] >= 0.80
                and result["summary"]["sketch"]["median_top16_recall"] >= 0.50
                and result["summary"]["sketch"]["valid_fraction"] == 1.0
                for result in results.values()
            ),
        },
        "F2_fisher_route_has_larger_behavioral_effect": {
            "supported": all(
                result["summary"]["routes"]["fisher_oracle"]["median_native_kl"]
                >= 1.25 * result["summary"]["routes"]["local_oracle"]["median_native_kl"]
                for result in results.values()
            ),
        },
        "F3_local_route_has_larger_query_effect": {
            "supported": all(
                result["summary"]["routes"]["local_oracle"]["median_absolute_direct_query_effect"]
                >= 1.25 * result["summary"]["routes"]["fisher_oracle"]["median_absolute_direct_query_effect"]
                for result in results.values()
            ),
        },
        "F4_local_and_fisher_supports_differ": {
            "supported": all(
                result["summary"]["routes"]["fisher_oracle"]["median_local_support_recall"] <= 0.50
                for result in results.values()
            ),
        },
        "F5_hierarchy_is_closer_to_local_target": {
            "supported": all(
                result["summary"]["routes"]["hierarchical128"]["median_fidelity_to_local_effect"]
                > result["summary"]["routes"]["hierarchical128"]["median_fidelity_to_fisher_effect"]
                for result in results.values()
            ),
        },
        "F6_fisher_ranking_predicts_finite_single_modes": {
            "supported": all(
                result["summary"]["finite_single_modes"]["median_paired_kl_ratio"] >= 2.0
                and result["summary"]["finite_single_modes"]["paired_win_rate"] >= 0.75
                for result in results.values()
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Local tensor-score versus downstream-Fisher mode oracles",
        "",
        "For each held-out state and natural output query, the existing local oracle ranks the query form's 512 eigenmodes by absolute scalar contribution. The new downstream oracle ranks the same modes by a Monte Carlo estimate of their Fisher-weighted final-logit effect after propagation through layers 9–17. Selection uses one 16-sample score-function sketch; an independent 16-sample sketch measures stability. Both routes receive the same 16-mode and 5%-input-norm intervention budget.",
        "",
        "Single-mode validation removes 10% of a mode's naturally present component. Each Fisher-top mode is paired with a non-top mode having the closest local query-contribution magnitude, so the control asks whether downstream weighting adds information beyond local amplitude.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Fisher oracle](layer8_downstream_fisher_oracle.png)",
        "",
        "| Corpus | Valid sketches; corr./top16 recall | Route | Native KL | |query effect| | Local/Fisher support | Fidelity to local/Fisher effect |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        summary = result["summary"]
        for route in ROUTES:
            row = summary["routes"][route]
            lines.append(
                f"| {corpus} | {summary['sketch']['valid_fraction']:.3f}; "
                f"{summary['sketch']['median_impact_correlation']:.3f}/"
                f"{summary['sketch']['median_top16_recall']:.3f} | {route} | "
                f"{row['median_native_kl']:.3e} | {row['median_absolute_direct_query_effect']:.3e} | "
                f"{row['median_local_support_recall']:.3f}/{row['median_fisher_support_recall']:.3f} | "
                f"{row['median_fidelity_to_local_effect']:.3f}/{row['median_fidelity_to_fisher_effect']:.3f} |"
            )
        finite = summary["finite_single_modes"]
        lines.append(
            f"| {corpus} | — | matched single-mode validation | KL ratio **{finite['median_paired_kl_ratio']:.2f}×** | "
            f"win rate **{finite['paired_win_rate']:.3f}** | — | — |"
        )
    lines += [
        "",
        "The downstream oracle is not a better semantic explanation by definition. It answers which locally available eigenmodes most affect the final distribution at this state. The local oracle answers which modes implement a specified layer-8 output query. A difference between them is an ontology result, not necessarily an error in either decomposition.",
        "",
        "## Estimator failure warning",
        "",
        "This first estimator uses two ordinary 16-draw Monte Carlo samples from the model distribution. At highly peaked predictions, all sampled tokens can be identical; centering then makes every estimated mode impact zero and the correlation undefined. Such rows are excluded from the displayed stability medians, but their Fisher-route selections are arbitrary and remain in the route aggregates. Therefore the downstream-oracle route table is exploratory and must not be treated as validated. The matched single-mode result is encouraging but will be rerun with exact top-token quadrature plus stratified tail sampling.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    corpora = list(results)
    x = np.arange(len(corpora))
    width = 0.35
    for index, route in enumerate(("local_oracle", "fisher_oracle")):
        axes[0].bar(x + (index - 0.5) * width, [
            results[c]["summary"]["routes"][route]["median_native_kl"] for c in corpora
        ], width, label=route)
        axes[1].bar(x + (index - 0.5) * width, [
            results[c]["summary"]["routes"][route]["median_absolute_direct_query_effect"] for c in corpora
        ], width, label=route)
    axes[0].set_title("Behavioral effect")
    axes[0].set_ylabel("median exact native KL")
    axes[1].set_title("Local query effect")
    axes[1].set_ylabel("median absolute direct effect")
    for axis in axes:
        axis.set_xticks(x, corpora)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
