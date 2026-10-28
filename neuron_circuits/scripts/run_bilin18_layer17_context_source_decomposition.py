#!/usr/bin/env python3
"""Decompose Layer-17 context-operator routing by exact architectural source."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_context_source_decomposition.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_source_decomposition.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_SOURCE_DECOMPOSITION.md")


@torch.inference_mode()
def collect_sources(model, spec, config, context_axes):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = np.asarray(config["positions"])
    position_tensor = torch.as_tensor(positions, device="cuda")
    source_rows, coefficient_rows, row_rows, position_rows = [], [], [], []
    dimension = model.config.n_embd
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x = x0
        value_bus = None
        for block in model.blocks[:17]:
            x, value_bus = block(x, value_bus, x0)
        block = model.blocks[17]
        carried = block.lambdas[0].float() * x.float()
        embedding = block.lambdas[1].float() * x0.float()
        mixed = carried + embedding
        attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        components = torch.stack(
            [carried[:, position_tensor], embedding[:, position_tensor],
             attention.float()[:, position_tensor]], dim=2)
        pre = components.sum(2)
        coefficients = F.rms_norm(pre, (dimension,)) @ context_axes
        batch = len(raw)
        source_rows.append(components.flatten(0, 1).cpu())
        coefficient_rows.append(coefficients.flatten(0, 1).cpu())
        row_rows.extend(np.repeat(np.arange(start, start + batch), len(positions)).tolist())
        position_rows.extend(np.tile(positions, batch).tolist())
    return {
        "sources": torch.cat(source_rows),
        "coefficients": torch.cat(coefficient_rows),
        "row": np.asarray(row_rows),
        "position": np.asarray(position_rows),
    }


def fit_means(data, selection, config):
    positions = config["positions"]
    means = torch.empty(len(positions), len(config["sources"]), data["sources"].shape[-1])
    for index, position in enumerate(positions):
        mask = selection & (data["position"] == position)
        means[index] = data["sources"][mask].mean(0)
    return means


def position_indices(position, positions):
    mapping = {value: index for index, value in enumerate(positions)}
    return torch.as_tensor([mapping[int(value)] for value in position], dtype=torch.long)


def gram_fidelity(target, prediction, gram):
    target = target.double()
    prediction = prediction.double()
    difference = target - prediction
    error = torch.einsum("bi,ij,bj->", difference, gram, difference)
    energy = torch.einsum("bi,ij,bj->", target, gram, target).clamp_min(1e-30)
    return 1.0 - float(error / energy)


def subset_coefficients(data, means, mask, context_axes, config):
    indices = position_indices(data["position"], config["positions"])
    baseline = means[indices]
    components = baseline.clone()
    for source in range(len(config["sources"])):
        if mask & (1 << source):
            components[:, source] = data["sources"][:, source]
    pre = components.sum(1).float().cuda()
    return (F.rms_norm(pre, (pre.shape[-1],)) @ context_axes).cpu()


def shapley(values, sources):
    allocations = {source: 0.0 for source in range(sources)}
    permutations = list(itertools.permutations(range(sources)))
    for permutation in permutations:
        mask = 0
        for source in permutation:
            before = values[mask]
            mask |= 1 << source
            allocations[source] += values[mask] - before
    return {source: value / len(permutations) for source, value in allocations.items()}


def evaluate(data, means, context_axes, gram, config):
    coefficients = {}
    values = {}
    for mask in range(1 << len(config["sources"])):
        coefficients[mask] = subset_coefficients(data, means, mask, context_axes, config)
        values[mask] = gram_fidelity(data["coefficients"], coefficients[mask], gram)
    allocation = shapley(values, len(config["sources"]))
    indices = position_indices(data["position"], config["positions"])
    centered = data["sources"] - means[indices]
    norm_stats = {}
    for source, name in enumerate(config["sources"]):
        norm_stats[name] = {
            "absolute_norm_median": float(data["sources"][:, source].norm(dim=1).median()),
            "variation_norm_median": float(centered[:, source].norm(dim=1).median()),
        }
    return {
        "subset_fidelity": {
            "+".join(config["sources"][source] for source in range(len(config["sources"]))
                     if mask & (1 << source)) or "mean": value
            for mask, value in values.items()
        },
        "shapley": {config["sources"][source]: value for source, value in allocation.items()},
        "removal_cost": {
            config["sources"][source]: 1.0 - values[((1 << len(config["sources"])) - 1) ^ (1 << source)]
            for source in range(len(config["sources"]))
        },
        "norms": norm_stats,
        "full_coefficient_relative_error": float(
            (coefficients[(1 << len(config["sources"])) - 1] - data["coefficients"]).double().norm()
            / data["coefficients"].double().norm().clamp_min(1e-30)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)
    flat = tensor.reshape(-1, config["rank"]).double()
    gram = flat.T @ flat

    print("collect discovery", flush=True)
    discovery = collect_sources(model, config["discovery"], config, context_axes)
    selection = discovery["row"] < config["discovery"]["rows"] // 2
    means = fit_means(discovery, selection, config)
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = collect_sources(model, spec, config, context_axes)
        results[corpus] = evaluate(data, means, context_axes, gram, config)

    full_mask_name = "+".join(config["sources"])
    embedding = "embedding"
    embedding_carried = "carried+embedding"
    exact_error = max(row["full_coefficient_relative_error"] for row in results.values())
    hypotheses = {
        "P0_three_sources_recompose_native_register": exact_error <= 1e-6,
        "P1_embedding_variation_is_a_strong_router": all(
            row["subset_fidelity"][embedding] >= .75 for row in results.values()),
        "P2_embedding_and_carried_state_are_nearly_sufficient": all(
            row["subset_fidelity"][embedding_carried] >= .95 for row in results.values()),
        "P3_attention_is_a_small_operator_correction": all(
            abs(row["shapley"]["attention"]) <= .10 for row in results.values()),
        "P4_embedding_is_the_largest_shapley_source": all(
            row["shapley"]["embedding"] == max(row["shapley"].values())
            for row in results.values()),
        "P5_attention_alone_is_not_a_router": all(
            row["subset_fidelity"]["attention"] <= .25 for row in results.values()),
        "P6_carried_state_adds_beyond_embedding": all(
            row["subset_fidelity"][embedding_carried]
            >= row["subset_fidelity"][embedding] + .05 for row in results.values()),
        "P7_embedding_dominance_transfers_ood": (
            max(row["subset_fidelity"][embedding] for row in results.values())
            - min(row["subset_fidelity"][embedding] for row in results.values()) <= .10),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "layer17_lambdas": [float(value) for value in model.blocks[17].lambdas.float().cpu()],
        "exact_error": exact_error,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Architectural sources of the Layer-17 context register", "",
        "The exact pre-normalization context state is split into carried state, direct original-embedding reinjection, and final attention. Position-conditioned mean filling preserves a common baseline before the native RMS normalization and tensor contraction.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Layer-17 carried/embedding coefficients: `{payload['layer17_lambdas'][0]:.4f}` / `{payload['layer17_lambdas'][1]:.4f}`. Full-source coefficient error: `{exact_error:.2e}`.",
              "", "## Source-subset operator fidelity", "",
              "| Corpus | mean | carried | embedding | attention | carried+embedding | embedding+attention | carried+attention | full |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        value = row["subset_fidelity"]
        lines.append(
            f"| {corpus} | {value['mean']:.3f} | {value['carried']:.3f} | "
            f"{value['embedding']:.3f} | {value['attention']:.3f} | "
            f"{value['carried+embedding']:.3f} | {value['embedding+attention']:.3f} | "
            f"{value['carried+attention']:.3f} | {value[full_mask_name]:.3f} |")
    lines += ["", "## Finite subset-Shapley allocation and removal cost", "",
              "| Corpus | Shapley carried / embedding / attention | Removal carried / embedding / attention |",
              "|---|---:|---:|"]
    for corpus, row in results.items():
        shapley = row["shapley"]; removal = row["removal_cost"]
        lines.append(
            f"| {corpus} | {shapley['carried']:.3f}/{shapley['embedding']:.3f}/{shapley['attention']:.3f} | "
            f"{removal['carried']:.3f}/{removal['embedding']:.3f}/{removal['attention']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "These are allocations of finite variation around position-conditioned discovery means. The carried source already contains computations and earlier embedding injections, so the rows are not exclusive storage percentages. They identify which source variation selects the final local operator at this boundary."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "lambdas": payload["layer17_lambdas"],
        "exact_error": exact_error,
        "summary": {corpus: {
            "subsets": row["subset_fidelity"],
            "shapley": row["shapley"],
            "removal": row["removal_cost"],
        } for corpus, row in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
