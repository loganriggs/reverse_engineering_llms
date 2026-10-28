#!/usr/bin/env python3
"""Finite causal validation of layer-8 oracle, flat, and hierarchical mode routes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_router_causal_validation.json")
OUTPUT = TENSOR_ROOT / "layer8_router_causal_validation.json"
REPORT = TENSOR_ROOT / "LAYER8_ROUTER_CAUSAL_VALIDATION.md"
FIGURE = TENSOR_ROOT / "layer8_router_causal_validation.png"
ROUTES = ("oracle", "flat256", "hierarchical128", "fixed16")


def pearson(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2 or first.std() < 1e-12 or second.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def route_indices(z, values, vectors, budget, coarse_dimension, refine_dimension, shortlist):
    """Choose query-specific modes using full, flat, hierarchical, and fixed routes."""
    full_projection = torch.einsum("d,qdk->qk", z, vectors)
    full_contribution = full_projection.square() * values
    oracle = full_contribution.abs().topk(budget, dim=-1).indices

    refine_projection = torch.einsum(
        "d,qdk->qk", z[:refine_dimension], vectors[:, :refine_dimension]
    )
    refine_contribution = refine_projection.square() * values
    flat = refine_contribution.abs().topk(budget, dim=-1).indices

    coarse_projection = torch.einsum(
        "d,qdk->qk", z[:coarse_dimension], vectors[:, :coarse_dimension]
    )
    coarse_contribution = coarse_projection.square() * values
    candidates = coarse_contribution.abs().topk(shortlist, dim=-1).indices
    refined_candidates = torch.gather(refine_contribution, -1, candidates)
    local = refined_candidates.abs().topk(budget, dim=-1).indices
    hierarchical = torch.gather(candidates, -1, local)
    fixed = torch.arange(budget, device=z.device)[None].expand(values.shape[0], -1)
    return {
        "oracle": oracle,
        "flat256": flat,
        "hierarchical128": hierarchical,
        "fixed16": fixed,
    }


@torch.inference_mode()
def prefix_to_layer(model, ids, target_layer):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    for block in model.blocks[:target_layer]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[target_layer]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    return x0, value_bus, pre, normalized


@torch.inference_mode()
def suffix_from_layer(model, target_layer, x0, value_bus, layer_output, position):
    dimension = model.config.n_embd
    x = layer_output
    for block in model.blocks[target_layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)


def repeat_batch(tensor, count):
    repeats = [count] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats)


def exact_kl(reference_logits, changed_logits) -> float:
    logp = F.log_softmax(reference_logits.float(), -1)
    p = logp.exp()
    logq = F.log_softmax(changed_logits.float(), -1)
    return float((p * (logp - logq)).sum())


@torch.inference_mode()
def native_recomposition_error(model, ids, layer, position) -> float:
    """Check that the split layer path equals an independent native block pass."""
    dimension = model.config.n_embd
    direct_x0 = F.rms_norm(model.embedding(ids), (dimension,))
    direct_x, direct_bus = direct_x0, None
    for block in model.blocks:
        direct_x, direct_bus = block(direct_x, direct_bus, direct_x0)
    direct_raw = model.lm_head(F.rms_norm(direct_x, (dimension,))[:, position]).float()
    direct_logits = 30 * torch.tanh(direct_raw / 30)

    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, layer)
    block = model.blocks[layer]
    write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    split_logits = suffix_from_layer(model, layer, x0, value_bus, pre + write, position)
    return float((direct_logits - split_logits).abs().max())


def select_sites(states, basis, values, vectors, count):
    z = states @ basis
    projection = torch.einsum("nd,qdk->nqk", z, vectors)
    scores = (projection.square() * values[None]).sum(-1)
    return scores.abs().topk(count, dim=0).indices.T.cpu().tolist()


def summarize(records):
    result = {}
    oracle_direct = [row["routes"]["oracle"]["direct_query_effect"] for row in records]
    for route in ROUTES:
        direct = [row["routes"][route]["direct_query_effect"] for row in records]
        per_query = []
        for query in sorted({row["query"] for row in records}):
            subset = [row for row in records if row["query"] == query]
            per_query.append(pearson(
                [row["routes"]["oracle"]["direct_query_effect"] for row in subset],
                [row["routes"][route]["direct_query_effect"] for row in subset],
            ))
        fidelity = [row["routes"][route]["fisher_fidelity_to_oracle"] for row in records]
        cosine = [row["routes"][route]["fisher_cosine_to_oracle"] for row in records]
        support = [row["routes"][route]["oracle_support_recall"] for row in records]
        result[route] = {
            "direct_query_effect_correlation": pearson(oracle_direct, direct),
            "median_per_query_direct_correlation": float(np.nanmedian(per_query)),
            "median_fisher_fidelity_to_oracle": float(np.median(fidelity)),
            "fisher_fidelity_q10": float(np.quantile(fidelity, 0.10)),
            "fisher_fidelity_q90": float(np.quantile(fidelity, 0.90)),
            "median_fisher_cosine_to_oracle": float(np.median(cosine)),
            "median_oracle_support_recall": float(np.median(support)),
            "support_fidelity_correlation": pearson(support, fidelity),
            "median_native_kl": float(np.median([
                row["routes"][route]["native_kl"] for row in records
            ])),
            "median_native_kl_ratio_to_oracle": float(np.median([
                row["routes"][route]["native_kl"]
                / max(row["routes"]["oracle"]["native_kl"], 1e-30)
                for row in records
            ])),
        }
    return result


@torch.inference_mode()
def evaluate_corpus(model, config, corpus_config, basis, directions, values, vectors):
    scan_config = {**config, "positions": [config["causal_position"]]}
    states, _ = collect(model, config["layer"], corpus_config, scan_config)
    selected = select_sites(states, basis, values, vectors, config["sites_per_query"])
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    records = []
    layer = config["layer"]
    position = config["causal_position"]
    block = model.blocks[layer]
    for query, query_sites in enumerate(selected):
        for row_index in query_sites:
            raw = np.asarray(rows[row_index, : position + 1], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")[None]
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, layer)
            x = normalized[0, position].float()
            z = x @ basis
            indices = route_indices(
                z, values, vectors, config["budget"], config["coarse_dimension"],
                config["refine_dimension"], config["shortlist"],
            )

            variants = normalized.repeat(len(ROUTES) + 1, 1, 1)
            oracle_norm = config["edit_norm_fraction"] * x.norm()
            for route_index, route in enumerate(ROUTES, start=1):
                mode_directions = basis @ vectors[query, :, indices[route][query]]
                raw_delta = -(x @ mode_directions) @ mode_directions.T
                delta = raw_delta * (oracle_norm / raw_delta.norm().clamp_min(1e-20))
                variants[route_index, position] = (x + delta).to(variants.dtype)
            product = block.mlp.Left(variants) * block.mlp.Right(variants)
            writes = block.mlp.Down(product) + block.mlp.Down_bias
            layer_outputs = repeat_batch(pre, len(ROUTES) + 1) + writes
            logits = suffix_from_layer(
                model, layer, repeat_batch(x0, len(ROUTES) + 1),
                repeat_batch(value_bus, len(ROUTES) + 1), layer_outputs, position,
            )
            native_logits = logits[0]
            native_write = writes[0, position].float()
            oracle_effect = logits[1] - native_logits
            oracle_set = set(indices["oracle"][query].cpu().tolist())
            route_rows = {}
            for route_index, route in enumerate(ROUTES, start=1):
                effect = logits[route_index] - native_logits
                fidelity, cosine, energy = fisher_fidelity(
                    native_logits, oracle_effect, effect,
                )
                route_set = set(indices[route][query].cpu().tolist())
                route_rows[route] = {
                    "direct_query_effect": float(
                        (writes[route_index, position].float() - native_write) @ directions[:, query]
                    ),
                    "fisher_fidelity_to_oracle": fidelity,
                    "fisher_cosine_to_oracle": cosine,
                    "oracle_effect_energy": energy,
                    "native_kl": exact_kl(native_logits, logits[route_index]),
                    "oracle_support_recall": len(route_set & oracle_set) / config["budget"],
                    "indices": sorted(route_set),
                }
            records.append({
                "query": query,
                "row": int(row_index),
                "routes": route_rows,
            })
        print("query", query, "complete", flush=True)
    return {"summary": summarize(records), "records": records}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    collection_config = {**config, "positions": config["discovery_positions"]}
    discovery_x, discovery_write = collect(
        model, config["layer"], config["discovery"], collection_config,
    )
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, :config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :config["queries"]]
    block = model.blocks[config["layer"]]
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)

    validation_source = np.load(config["discovery"]["path"], mmap_mode="r")
    validation_ids = torch.as_tensor(
        np.asarray(validation_source[config["discovery"]["start"], :config["causal_position"] + 1], dtype=np.int64),
        device="cuda",
    )[None]
    recomposition_error = native_recomposition_error(
        model, validation_ids, config["layer"], config["causal_position"],
    )
    if recomposition_error > 1e-5:
        raise AssertionError(f"native split-path recomposition error {recomposition_error:.3e}")

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("causal evaluation", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, directions, values, vectors,
        )

    fine = results["fineweb"]["summary"]
    wiki = results["wikitext"]["summary"]
    hypotheses = {
        "C1_hierarchical_direct_effect": {
            "supported": all(results[c]["summary"]["hierarchical128"]["direct_query_effect_correlation"] >= 0.85 for c in results),
        },
        "C2_hierarchical_downstream_fidelity": {
            "supported": all(results[c]["summary"]["hierarchical128"]["median_fisher_fidelity_to_oracle"] >= 0.50 for c in results),
        },
        "C3_hierarchical_near_flat": {
            "supported": all(
                results[c]["summary"]["flat256"]["median_fisher_fidelity_to_oracle"]
                - results[c]["summary"]["hierarchical128"]["median_fisher_fidelity_to_oracle"] <= 0.10
                for c in results
            ),
        },
        "C4_hierarchical_beats_fixed": {
            "supported": all(
                results[c]["summary"]["hierarchical128"]["median_fisher_fidelity_to_oracle"]
                - results[c]["summary"]["fixed16"]["median_fisher_fidelity_to_oracle"] >= 0.20
                for c in results
            ),
        },
        "C5_causal_ood_stability": {
            "supported": (
                abs(fine["hierarchical128"]["direct_query_effect_correlation"] - wiki["hierarchical128"]["direct_query_effect_correlation"]) <= 0.10
                and abs(fine["hierarchical128"]["median_fisher_fidelity_to_oracle"] - wiki["hierarchical128"]["median_fisher_fidelity_to_oracle"]) <= 0.10
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "native_recomposition_max_abs_error": recomposition_error,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Finite causal validation of the layer-8 mode router",
        "",
        "For each of 16 natural write queries, six held-out final-position states with the largest absolute full-form score were selected. Each route chose 16 input eigenmodes. The selected 16D component was removed and rescaled to the same 5% input-state norm, the layer-8 MLP was recomputed, and all layers 9–17 plus the unembedding were rerun. The target is the per-example full-information oracle route, not the unedited model.",
        "",
        "The hierarchical shortlist-128 setting was selected after the observational sweep; the causal thresholds were locked before this run. Consequently this is prospective for causal behavior but not a pristine prospective selection of the architecture.",
        f"The independently executed native block path and split layer-8 path agree to maximum absolute logit error `{recomposition_error:.3e}`.",
        "",
        "## Locked causal hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Causal router validation](layer8_router_causal_validation.png)",
        "",
        "| Corpus | Route | Direct effect corr. | Median per-query corr. | Fisher fidelity/cosine | Oracle support recall | Native KL | KL/oracle |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_result in results.items():
        for route in ROUTES:
            row = corpus_result["summary"][route]
            lines.append(
                f"| {corpus} | {route} | {row['direct_query_effect_correlation']:.3f} | "
                f"{row['median_per_query_direct_correlation']:.3f} | "
                f"{row['median_fisher_fidelity_to_oracle']:.3f}/{row['median_fisher_cosine_to_oracle']:.3f} | "
                f"{row['median_oracle_support_recall']:.3f} | {row['median_native_kl']:.3e} | "
                f"{row['median_native_kl_ratio_to_oracle']:.3f} |"
            )
    lines += [
        "",
        "Fisher fidelity is 1 for an exact match to the oracle logit effect, 0 when the squared softmax-local error equals the oracle effect energy, and can be negative. It measures agreement with this particular oracle intervention, not whether a route is independently useful or semantically meaningful.",
        "",
        "## Red-team notes",
        "",
        "The median hides substantial site and query heterogeneity. Hierarchical fidelity has a 10th–90th percentile range of "
        f"{results['fineweb']['summary']['hierarchical128']['fisher_fidelity_q10']:.3f}–{results['fineweb']['summary']['hierarchical128']['fisher_fidelity_q90']:.3f} on FineWeb and "
        f"{results['wikitext']['summary']['hierarchical128']['fisher_fidelity_q10']:.3f}–{results['wikitext']['summary']['hierarchical128']['fisher_fidelity_q90']:.3f} on WikiText. "
        "Support recall and causal fidelity are positively but imperfectly related: their correlations are "
        f"{results['fineweb']['summary']['hierarchical128']['support_fidelity_correlation']:.3f} and "
        f"{results['wikitext']['summary']['hierarchical128']['support_fidelity_correlation']:.3f}. Mode identity alone therefore does not determine downstream equivalence.",
        "",
        "Every edit was norm-matched, which prevents a trivial intervention-size advantage but means this is a directional-subspace test rather than literal removal of each route's naturally sized component. Sites were selected for large query scores, so the result applies to high-use sites and should not be read as an average-token estimate. The oracle is also query-local: it chooses modes by immediate layer-8 contribution, not by final-logit importance. A downstream-aware oracle could differ.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(ROUTES))
    width = 0.36
    for corpus_index, (corpus, corpus_result) in enumerate(results.items()):
        offset = (corpus_index - 0.5) * width
        axes[0].bar(x + offset, [corpus_result["summary"][r]["direct_query_effect_correlation"] for r in ROUTES], width, label=corpus)
        axes[1].bar(x + offset, [corpus_result["summary"][r]["median_fisher_fidelity_to_oracle"] for r in ROUTES], width, label=corpus)
    axes[0].set_title("Direct query effect")
    axes[0].set_ylabel("correlation with oracle")
    axes[1].set_title("Downstream distribution effect")
    axes[1].set_ylabel("median Fisher fidelity to oracle")
    for axis in axes:
        axis.set_xticks(x, ROUTES, rotation=20)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
