#!/usr/bin/env python3
"""Directly test cancellation among Fisher-selected layer-8 modes."""

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
    prefix_to_layer,
    repeat_batch,
    suffix_from_layer,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_fisher_cancellation_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_fisher_cancellation_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_FISHER_CANCELLATION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_fisher_cancellation_redteam.png"


def fisher_energy(reference_logits, effect):
    probability = F.softmax(reference_logits.float(), -1)
    centered = effect.float() - (probability * effect.float()).sum()
    return float((probability * centered.square()).sum())


@torch.inference_mode()
def finite_individual_and_joint(model, pre, normalized, x0, value_bus, position,
                                mode_directions, mode_indices, removal_fraction):
    block = model.blocks[8]
    x = normalized[0, position].float()
    projections = x @ mode_directions[:, mode_indices]
    individual_deltas = -removal_fraction * projections[:, None] * mode_directions[:, mode_indices].T
    joint_delta = individual_deltas.sum(0)
    variants = normalized.repeat(len(mode_indices) + 2, 1, 1)
    for index, delta in enumerate(individual_deltas, start=1):
        variants[index, position] = (x + delta).to(variants.dtype)
    variants[-1, position] = (x + joint_delta).to(variants.dtype)
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(variants)) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(variants)),
        repeat_batch(value_bus, len(variants)), outputs, position,
    )
    native = logits[0]
    individual_effects = logits[1:-1] - native
    joint_effect = logits[-1] - native
    signed_sum = individual_effects.sum(0)
    individual_energies = [fisher_energy(native, effect) for effect in individual_effects]
    sum_individual_energy = sum(individual_energies)
    sum_effect_energy = fisher_energy(native, signed_sum)
    joint_energy = fisher_energy(native, joint_effect)
    fidelity, cosine, _ = fisher_fidelity(native, joint_effect, signed_sum)
    return {
        "sum_individual_fisher_energy": sum_individual_energy,
        "signed_sum_fisher_energy": sum_effect_energy,
        "exact_joint_fisher_energy": joint_energy,
        "signed_sum_cancellation_ratio": sum_effect_energy / max(sum_individual_energy, 1e-30),
        "exact_joint_cancellation_ratio": joint_energy / max(sum_individual_energy, 1e-30),
        "sum_to_joint_fisher_fidelity": fidelity,
        "sum_to_joint_fisher_cosine": cosine,
        "sum_individual_kl": sum(exact_kl(native, logits[index]) for index in range(1, len(logits) - 1)),
        "exact_joint_kl": exact_kl(native, logits[-1]),
        "joint_kl_to_sum_individual_kl": exact_kl(native, logits[-1])
        / max(sum(exact_kl(native, logits[index]) for index in range(1, len(logits) - 1)), 1e-30),
        "joint_input_norm": float(joint_delta.norm()),
        "root_sum_individual_input_norm": float(individual_deltas.square().sum().sqrt()),
    }


def summarize(records, routes):
    result = {}
    for route in routes:
        rows = [record["routes"][route] for record in records]
        result[route] = {
            key: float(np.median([row[key] for row in rows]))
            for key in (
                "signed_sum_cancellation_ratio",
                "exact_joint_cancellation_ratio",
                "sum_to_joint_fisher_fidelity",
                "sum_to_joint_fisher_cosine",
                "joint_kl_to_sum_individual_kl",
                "exact_joint_kl",
            )
        }
    return result


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    discovery_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, discovery_write = collect(
        model, 8, source_config["discovery"], discovery_config,
    )
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, :source_config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :source_config["queries"]]
    block = model.blocks[8]
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    _, vectors = ordered_eigensystem(matrices)

    results = {}
    for corpus, corpus_result in source["results"].items():
        corpus_config = source_config["evaluation"][corpus]
        corpus_source = np.load(corpus_config["path"], mmap_mode="r")
        rows = corpus_source[
            corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]
        ]
        records = []
        for record in corpus_result["records"]:
            query = record["query"]
            row_index = record["row"]
            ids = torch.as_tensor(
                np.asarray(rows[row_index, :source_config["causal_position"] + 1], dtype=np.int64),
                device="cuda",
            )[None]
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
            mode_directions = basis @ vectors[query]
            route_rows = {}
            for route in config["routes"]:
                indices = torch.as_tensor(
                    record["routes"][route]["indices"], device="cuda", dtype=torch.long,
                )
                route_rows[route] = finite_individual_and_joint(
                    model, pre, normalized, x0, value_bus,
                    source_config["causal_position"], mode_directions, indices,
                    config["component_removal_fraction"],
                )
            records.append({"query": query, "row": row_index, "routes": route_rows})
        results[corpus] = {"summary": summarize(records, config["routes"]), "records": records}
        print("completed", corpus, flush=True)

    fine_ratio = results["fineweb"]["summary"]["fisher_oracle"]["exact_joint_cancellation_ratio"]
    wiki_ratio = results["wikitext"]["summary"]["fisher_oracle"]["exact_joint_cancellation_ratio"]
    hypotheses = {
        "C1_fisher_individual_effects_cancel": {
            "supported": all(
                results[c]["summary"]["fisher_oracle"]["signed_sum_cancellation_ratio"] <= 0.25
                for c in results
            ),
        },
        "C2_exact_joint_effect_remains_cancelled": {
            "supported": all(
                results[c]["summary"]["fisher_oracle"]["exact_joint_cancellation_ratio"] <= 0.25
                for c in results
            ),
        },
        "C3_joint_nonlinearity_is_not_the_explanation": {
            "supported": all(
                results[c]["summary"][r]["sum_to_joint_fisher_fidelity"] >= 0.90
                for c in results for r in config["routes"]
            ),
        },
        "C4_fisher_set_cancels_more_than_local_set": {
            "supported": all(
                results[c]["summary"]["fisher_oracle"]["exact_joint_cancellation_ratio"]
                <= 0.5 * results[c]["summary"]["local_oracle"]["exact_joint_cancellation_ratio"]
                for c in results
            ),
        },
        "C5_cancellation_is_ood_stable": {
            "supported": abs(fine_ratio - wiki_ratio) <= 0.10,
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Cancellation red-team for downstream-Fisher modes",
        "",
        "For each frozen local or Fisher top-16 set, every mode is removed individually by 10% of its naturally present component. The same 16 input deltas are then applied jointly. This compares the sum of individual downstream effects with the exact joint forward pass, without renormalizing the joint input direction.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Cancellation red-team](layer8_fisher_cancellation_redteam.png)",
        "",
        "| Corpus | Route | Signed-sum energy / sum energies | Exact-joint energy / sum energies | Sum→joint fidelity/cosine | Joint KL / sum KL | Joint KL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for route in config["routes"]:
            row = result["summary"][route]
            lines.append(
                f"| {corpus} | {route} | {row['signed_sum_cancellation_ratio']:.3f} | "
                f"{row['exact_joint_cancellation_ratio']:.3f} | "
                f"{row['sum_to_joint_fisher_fidelity']:.3f}/{row['sum_to_joint_fisher_cosine']:.3f} | "
                f"{row['joint_kl_to_sum_individual_kl']:.3f} | {row['exact_joint_kl']:.3e} |"
            )
    lines += [
        "",
        "A ratio below one means signed downstream effects cancel. It does not mean the modes are unnecessary: each was selected for a large individual effect. High sum-to-joint fidelity would locate the small joint effect in destructive addition rather than nonlinear failure of the individual-effect approximation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    corpora = list(results)
    x = np.arange(len(corpora))
    width = 0.35
    for index, route in enumerate(config["routes"]):
        axes[0].bar(x + (index - 0.5) * width, [
            results[c]["summary"][route]["exact_joint_cancellation_ratio"] for c in corpora
        ], width, label=route)
        axes[1].bar(x + (index - 0.5) * width, [
            results[c]["summary"][route]["sum_to_joint_fisher_fidelity"] for c in corpora
        ], width, label=route)
    axes[0].set_title("Signed cancellation")
    axes[0].set_ylabel("joint energy / sum individual energy")
    axes[1].set_title("Nonlinear joint correction")
    axes[1].set_ylabel("sum-to-joint Fisher fidelity")
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
