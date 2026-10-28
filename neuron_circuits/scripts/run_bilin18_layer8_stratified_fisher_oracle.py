#!/usr/bin/env python3
"""Stratified downstream-Fisher oracle with exact top-token quadrature."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_downstream_fisher_oracle import (
    ROUTES,
    finite_routes,
    finite_single_modes,
    matched_local_controls,
    mode_write_tangents,
    summarize,
)
from run_bilin18_layer8_router_causal_validation import (
    fisher_fidelity,
    pearson,
    prefix_to_layer,
    route_indices,
    select_sites,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_stratified_fisher_oracle.json")
OUTPUT = TENSOR_ROOT / "layer8_stratified_fisher_oracle.json"
REPORT = TENSOR_ROOT / "LAYER8_STRATIFIED_FISHER_ORACLE.md"
FIGURE = TENSOR_ROOT / "layer8_stratified_fisher_oracle.png"


def weighted_fisher_impact(gradients, weights, mode_write_tangents):
    response = gradients @ mode_write_tangents.T
    return (weights[:, None] * response.square()).sum(0)


def stratified_fisher_sketch(model, base_state, x0, value_bus, position,
                             exact_top, tail_samples, seed):
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
    probabilities = logp.exp().detach()[0]
    top_probability, top_tokens = probabilities.topk(exact_top)
    tail_probability = probabilities.clone()
    tail_probability[top_tokens] = 0
    tail_mass = tail_probability.sum()
    conditional_tail = tail_probability / tail_mass.clamp_min(1e-30)
    generator = torch.Generator(device=base_state.device).manual_seed(seed)
    tail_tokens = torch.multinomial(
        conditional_tail, 2 * tail_samples, replacement=True, generator=generator,
    )
    all_tokens = torch.cat((top_tokens, tail_tokens))
    unique_tokens, inverse = torch.unique(all_tokens, return_inverse=True)
    unique_gradients = []
    for index, token in enumerate(unique_tokens):
        unique_gradients.append(torch.autograd.grad(
            logp[0, token], delta, retain_graph=index + 1 < len(unique_tokens),
        )[0].detach())
    gradients = torch.stack(unique_gradients)[inverse]
    top_gradients = gradients[:exact_top]
    first_tail = gradients[exact_top:exact_top + tail_samples]
    second_tail = gradients[exact_top + tail_samples:]
    tail_weights = torch.full(
        (tail_samples,), float(tail_mass) / tail_samples,
        device=base_state.device,
    )
    weights = torch.cat((top_probability, tail_weights))
    return (
        torch.cat((top_gradients, first_tail)), weights,
        torch.cat((top_gradients, second_tail)), weights,
        {
            "top_probability_mass": float(top_probability.sum()),
            "tail_probability_mass": float(tail_mass),
            "unique_tail_tokens": int(torch.unique(tail_tokens).numel()),
        },
    )


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
                mode_tangents, amplitudes = mode_write_tangents(block, x, mode_directions)
            seed = config["seed"] + corpus_index * 100000 + query * 100 + site_number
            first_g, first_w, second_g, second_w, metadata = stratified_fisher_sketch(
                model, base_state, x0.detach(), value_bus.detach(), position,
                config["exact_top_tokens"], config["tail_samples_per_split"], seed,
            )
            first_impact = weighted_fisher_impact(first_g, first_w, mode_tangents)
            second_impact = weighted_fisher_impact(second_g, second_w, mode_tangents)
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
            records.append({
                "query": query,
                "row": int(row_index),
                "sketch_valid": True,
                "sketch_impact_correlation": pearson(
                    first_impact.cpu().numpy(), second_impact.cpu().numpy(),
                ),
                "sketch_top16_recall": len(
                    set(fisher_indices.cpu().tolist()) & set(second_indices.cpu().tolist())
                ) / config["budget"],
                "sketch_metadata": metadata,
                "routes": route_rows,
                "single_mode_top_kl": single_kl[:config["finite_modes_per_site"]],
                "single_mode_control_kl": single_kl[config["finite_modes_per_site"]:],
            })
            del first_g, second_g
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
        discovery_x, discovery_write = collect(model, 8, config["discovery"], discovery_config)
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
        print("stratified Fisher", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, directions, values, vectors, corpus_index,
        )

    def route(corpus, name):
        return results[corpus]["summary"]["routes"][name]

    corpora = list(results)
    r2_supported = all(
        results[c]["summary"]["finite_single_modes"]["median_paired_kl_ratio"] >= 2
        and results[c]["summary"]["finite_single_modes"]["paired_win_rate"] >= 0.75
        for c in corpora
    )
    hypotheses = {
        "R1_stratified_sketch_is_stable": {
            "supported": all(
                results[c]["summary"]["sketch"]["median_impact_correlation"] >= 0.90
                and results[c]["summary"]["sketch"]["median_top16_recall"] >= 0.75
                for c in corpora
            ),
        },
        "R2_fisher_modes_pass_matched_finite_control": {"supported": r2_supported},
        "R3_fisher_set_has_larger_combined_effect": {
            "supported": all(
                route(c, "fisher_oracle")["median_native_kl"]
                >= 1.25 * route(c, "local_oracle")["median_native_kl"]
                for c in corpora
            ),
        },
        "R4_individual_importance_cancels_in_combination": {
            "supported": r2_supported and all(
                route(c, "fisher_oracle")["median_native_kl"]
                <= 0.5 * route(c, "local_oracle")["median_native_kl"]
                for c in corpora
            ),
        },
        "R5_local_and_fisher_supports_differ": {
            "supported": all(
                route(c, "fisher_oracle")["median_local_support_recall"] <= 0.50
                for c in corpora
            ),
        },
        "R6_hierarchy_targets_local_computation": {
            "supported": all(
                route(c, "hierarchical128")["median_fidelity_to_local_effect"]
                > route(c, "hierarchical128")["median_fidelity_to_fisher_effect"]
                for c in corpora
            ),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Stratified downstream-Fisher oracle",
        "",
        "This repairs the failed ordinary-Monte-Carlo pilot. Each suffix Fisher estimate sums exact probability-weighted score gradients for the top 16 output tokens, then estimates the remaining probability tail with 16 conditional samples. Two independent tail samples test stability. Selection uses the first estimate; the second and finite interventions are held-out checks.",
        "",
        "## Locked post-pilot hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Stratified Fisher oracle](layer8_stratified_fisher_oracle.png)",
        "",
        "| Corpus | Sketch corr./top16 recall | Route | Native KL | |query effect| | Local/Fisher support | Fidelity to local/Fisher effect |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        summary = result["summary"]
        for route_name in ROUTES:
            row = summary["routes"][route_name]
            lines.append(
                f"| {corpus} | {summary['sketch']['median_impact_correlation']:.3f}/"
                f"{summary['sketch']['median_top16_recall']:.3f} | {route_name} | "
                f"{row['median_native_kl']:.3e} | {row['median_absolute_direct_query_effect']:.3e} | "
                f"{row['median_local_support_recall']:.3f}/{row['median_fisher_support_recall']:.3f} | "
                f"{row['median_fidelity_to_local_effect']:.3f}/{row['median_fidelity_to_fisher_effect']:.3f} |"
            )
        finite = summary["finite_single_modes"]
        top_mass = np.median([r["sketch_metadata"]["top_probability_mass"] for r in result["records"]])
        lines.append(
            f"| {corpus} | top-token probability mass {top_mass:.3f} | matched single modes | "
            f"KL ratio **{finite['median_paired_kl_ratio']:.2f}×** | win **{finite['paired_win_rate']:.3f}** | — | — |"
        )
    lines += [
        "",
        "Local and downstream importance are different objectives. In particular, a set of modes chosen for large individual Fisher effects need not have a large joint effect: their logit changes can cancel, and the matched-norm joint projection changes their natural component amplitudes. The single-mode and combined-route columns must therefore be read together.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(corpora))
    width = 0.35
    for index, name in enumerate(("local_oracle", "fisher_oracle")):
        axes[0].bar(x + (index - 0.5) * width, [route(c, name)["median_native_kl"] for c in corpora], width, label=name)
        axes[1].bar(x + (index - 0.5) * width, [route(c, name)["median_absolute_direct_query_effect"] for c in corpora], width, label=name)
    axes[0].set_title("Joint behavioral effect")
    axes[0].set_ylabel("median exact KL")
    axes[1].set_title("Local query effect")
    axes[1].set_ylabel("median absolute effect")
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
