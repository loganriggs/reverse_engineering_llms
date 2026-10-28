#!/usr/bin/env python3
"""Recompute final attention after Layer-17 embedding/carried source edits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_context_source_mediation_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_source_mediation_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_CONTEXT_SOURCE_MEDIATION_REDTEAM.md")


@torch.inference_mode()
def prefix(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    for block in model.blocks[:17]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[17]
    carried = block.lambdas[0].float() * x.float()
    embedding = block.lambdas[1].float() * x0.float()
    return carried, embedding, value_bus


@torch.inference_mode()
def fit_means(model, spec, config):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    positions = config["positions"]
    carried_rows = [[] for _ in positions]
    embedding_rows = [[] for _ in positions]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        carried, embedding, _ = prefix(model, ids)
        for index, position in enumerate(positions):
            carried_rows[index].append(carried[:, position].cpu())
            embedding_rows[index].append(embedding[:, position].cpu())
    return {
        "carried": torch.stack([torch.cat(rows).mean(0) for rows in carried_rows]).cuda(),
        "embedding": torch.stack([torch.cat(rows).mean(0) for rows in embedding_rows]).cuda(),
    }


def coefficients(pre, context_axes):
    return F.rms_norm(pre.float(), (pre.shape[-1],)) @ context_axes


@torch.inference_mode()
def collect(model, spec, config, context_axes, means):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    names = [
        "native", "embedding_mean_held", "embedding_mean_recomputed",
        "embedding_cycle_held", "embedding_cycle_recomputed",
        "carried_mean_held", "carried_mean_recomputed",
        "carried_cycle_held", "carried_cycle_recomputed",
    ]
    output = {name: [] for name in names}
    exact_error = 0.0
    exact_energy = 0.0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        carried, embedding, value_bus = prefix(model, ids)
        mixed = carried + embedding
        block = model.blocks[17]
        native_attention, _ = block.attn(F.rms_norm(mixed, (mixed.shape[-1],)), value_bus)
        native_attention = native_attention.float()
        _, reference_state = prefix_to_layer17(model, ids)
        reference_coefficients = reference_state.float() @ context_axes
        batch = len(ids)
        for index, position in enumerate(config["positions"]):
            native_pre = mixed[:, position] + native_attention[:, position]
            native_coefficients = coefficients(native_pre, context_axes)
            reference = reference_coefficients[:, position]
            exact_error += float((native_coefficients.double() - reference.double()).square().sum())
            exact_energy += float(reference.double().square().sum())
            output["native"].append(native_coefficients.cpu())

            variants = mixed.repeat(4, 1, 1)
            variants[0 * batch:1 * batch, position] = (
                carried[:, position] + means["embedding"][index])
            variants[1 * batch:2 * batch, position] = (
                carried[:, position] + torch.roll(embedding[:, position], 1, 0))
            variants[2 * batch:3 * batch, position] = (
                means["carried"][index] + embedding[:, position])
            variants[3 * batch:4 * batch, position] = (
                torch.roll(carried[:, position], 1, 0) + embedding[:, position])
            repeat = [4] + [1] * (value_bus.ndim - 1)
            variant_attention, _ = block.attn(
                F.rms_norm(variants, (variants.shape[-1],)), value_bus.repeat(*repeat))
            variant_attention = variant_attention.float()

            edits = {
                "embedding_mean": variants[0 * batch:1 * batch],
                "embedding_cycle": variants[1 * batch:2 * batch],
                "carried_mean": variants[2 * batch:3 * batch],
                "carried_cycle": variants[3 * batch:4 * batch],
            }
            for variant_index, (name, state) in enumerate(edits.items()):
                held_pre = state[:, position] + native_attention[:, position]
                recomputed_pre = (
                    state[:, position]
                    + variant_attention[variant_index * batch:(variant_index + 1) * batch, position])
                output[f"{name}_held"].append(coefficients(held_pre, context_axes).cpu())
                output[f"{name}_recomputed"].append(coefficients(recomputed_pre, context_axes).cpu())
    return ({name: torch.cat(values) for name, values in output.items()},
            float(np.sqrt(exact_error / max(exact_energy, 1e-30))))


def gram_fidelity(target, prediction, gram):
    target = target.double(); prediction = prediction.double(); difference = target - prediction
    error = torch.einsum("bi,ij,bj->", difference, gram, difference)
    energy = torch.einsum("bi,ij,bj->", target, gram, target).clamp_min(1e-30)
    return 1.0 - float(error / energy)


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

    print("fit source means", flush=True)
    means = fit_means(model, config["discovery"], config)
    results = {}
    exact_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        rows, corpus_exact_error = collect(model, spec, config, context_axes, means)
        target = rows.pop("native")
        metrics = {name: gram_fidelity(target, prediction, gram) for name, prediction in rows.items()}
        results[corpus] = metrics
        exact_error = max(exact_error, corpus_exact_error)

    embedding_cost = lambda row, suffix: 1 - row[f"embedding_mean_{suffix}"]
    carried_cost = lambda row, suffix: 1 - row[f"carried_mean_{suffix}"]
    hypotheses = {
        "M0_native_recomposition_is_exact": exact_error <= 1e-6,
        "M1_direct_embedding_mean_replacement_is_small": all(
            row["embedding_mean_recomputed"] >= .98 for row in results.values()),
        "M2_direct_embedding_cycle_is_small": all(
            row["embedding_cycle_recomputed"] >= .95 for row in results.values()),
        "M3_embedding_attention_mediation_is_small": all(
            abs(row["embedding_mean_recomputed"] - row["embedding_mean_held"]) <= .02
            and abs(row["embedding_cycle_recomputed"] - row["embedding_cycle_held"]) <= .02
            for row in results.values()),
        "M4_carried_mean_replacement_is_material": all(
            row["carried_mean_recomputed"] <= .80 for row in results.values()),
        "M5_carried_cycle_is_material": all(
            row["carried_cycle_recomputed"] <= .90 for row in results.values()),
        "M6_carried_cost_dominates_embedding_cost": all(
            carried_cost(row, "recomputed") >= 10 * embedding_cost(row, "recomputed")
            for row in results.values()),
        "M7_attention_recomputation_preserves_source_order": all(
            carried_cost(row, suffix) > embedding_cost(row, suffix)
            for row in results.values() for suffix in ("held", "recomputed")),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {"config": config, "hypotheses": hypotheses,
               "native_recomposition_error": exact_error, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Attention-mediated red team for Layer-17 context sources", "",
        "Selected-position embedding or carried terms are mean-replaced or cyclically substituted. Final attention is either held native or exactly recomputed with the edited mixed state and native persistent value bus.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Operator fidelity", "",
              "| Corpus | embedding mean held/recomputed | embedding cycle held/recomputed | carried mean held/recomputed | carried cycle held/recomputed |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {row['embedding_mean_held']:.6f}/{row['embedding_mean_recomputed']:.6f} | "
            f"{row['embedding_cycle_held']:.6f}/{row['embedding_cycle_recomputed']:.6f} | "
            f"{row['carried_mean_held']:.3f}/{row['carried_mean_recomputed']:.3f} | "
            f"{row['carried_cycle_held']:.3f}/{row['carried_cycle_recomputed']:.3f} |")
    lines += ["", "The intervention changes only the current Layer-17 injection boundary. A negligible direct embedding effect is compatible with strong embedding predictability because the carried state has already compiled token identity through seventeen earlier blocks and embedding injections."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
