#!/usr/bin/env python3
"""Decompose full token edits across the tensor transformer's persistent carriers.

At every block boundary, propagate all eight hybrids of the changed/base
residual sequence, original embedding stream, and persistent Layer-0 value bus
through the untouched suffix. Exact logit-space Shapley effects expose both
unique contributions and interactions without interpreting any carrier as a
literal percentage of stored information.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_carrier_depth_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_carrier_depth_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_CARRIER_DEPTH_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_boundary_carrier_depth_audit.png"
FACTORS = ("residual_sequence", "embedding_stream", "value_bus")
ROUTES = {
    "base": 0,
    "residual_only": 1,
    "embedding_only": 2,
    "value_bus_only": 4,
    "residual_embedding": 3,
    "residual_bus": 5,
    "embedding_bus": 6,
    "full": 7,
}


def fisher_terms(reference, target, prediction):
    """Per-example Fisher energy, prediction energy, cross, and squared error."""
    probability = F.softmax(reference.float(), -1)
    target = target.float() - (probability * target.float()).sum(-1, keepdim=True)
    prediction = prediction.float() - (probability * prediction.float()).sum(-1, keepdim=True)
    target_energy = (probability * target.square()).sum(-1)
    prediction_energy = (probability * prediction.square()).sum(-1)
    cross = (probability * target * prediction).sum(-1)
    error = (probability * (prediction - target).square()).sum(-1)
    return target_energy, prediction_energy, cross, error


def exact_kl(reference, changed):
    logp = F.log_softmax(reference.float(), -1)
    return (logp.exp() * (logp - F.log_softmax(changed.float(), -1))).sum(-1)


def shapley_effects(world_logits):
    """Exact three-player vector Shapley effects; shape is [8,B,V]."""
    result = {}
    count = len(FACTORS)
    for factor_index, factor in enumerate(FACTORS):
        bit = 1 << factor_index
        effect = torch.zeros_like(world_logits[0])
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size) * math.factorial(count - size - 1)
                / math.factorial(count)
            )
            effect += weight * (world_logits[mask | bit] - world_logits[mask])
        result[factor] = effect
    return result


@torch.inference_mode()
def trace(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    boundaries = []
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
        boundaries.append({"residual_sequence": x, "embedding_stream": x0, "value_bus": value_bus})
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return boundaries, 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def suffix_worlds(model, layer, base, changed):
    batch = base["residual_sequence"].shape[0]
    worlds = []
    for mask in range(8):
        worlds.append({
            factor: (changed[factor] if mask & (1 << index) else base[factor])
            for index, factor in enumerate(FACTORS)
        })
    x = torch.cat([world["residual_sequence"] for world in worlds])
    x0 = torch.cat([world["embedding_stream"] for world in worlds])
    value_bus = torch.cat([world["value_bus"] for world in worlds])
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    dimension = model.config.n_embd
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    logits = 30 * torch.tanh(raw / 30)
    return logits.reshape(8, batch, -1)


def empty_accumulator():
    return {
        "count": 0,
        "target_energy": 0.0,
        "forward_full_kl": 0.0,
        "reverse_full_kl": 0.0,
        "max_base_error": 0.0,
        "max_changed_error": 0.0,
        "routes": {
            name: {"prediction_energy": 0.0, "cross": 0.0, "error": 0.0,
                   "forward_kl": 0.0, "reverse_error_kl": 0.0}
            for name in ROUTES if name != "base"
        },
        "shapley_cross": {factor: 0.0 for factor in FACTORS},
        "per_example": {factor: [] for factor in FACTORS},
    }


def update_accumulator(acc, worlds, native_base, native_changed):
    base, changed = worlds[0], worlds[7]
    target = changed - base
    target_energy, _, _, _ = fisher_terms(base, target, target)
    acc["count"] += len(base)
    acc["target_energy"] += float(target_energy.sum())
    acc["forward_full_kl"] += float(exact_kl(base, changed).sum())
    acc["reverse_full_kl"] += float(exact_kl(changed, base).sum())
    acc["max_base_error"] = max(acc["max_base_error"], float((base - native_base).abs().max()))
    acc["max_changed_error"] = max(acc["max_changed_error"], float((changed - native_changed).abs().max()))
    for name, mask in ROUTES.items():
        if name == "base":
            continue
        prediction = worlds[mask] - base
        _, prediction_energy, cross, error = fisher_terms(base, target, prediction)
        row = acc["routes"][name]
        row["prediction_energy"] += float(prediction_energy.sum())
        row["cross"] += float(cross.sum())
        row["error"] += float(error.sum())
        row["forward_kl"] += float(exact_kl(base, worlds[mask]).sum())
        row["reverse_error_kl"] += float(exact_kl(changed, worlds[mask]).sum())
    for factor, effect in shapley_effects(worlds).items():
        _, _, cross, _ = fisher_terms(base, target, effect)
        acc["shapley_cross"][factor] += float(cross.sum())
        acc["per_example"][factor].extend((cross / target_energy.clamp_min(1e-30)).cpu().tolist())


def finalize(acc):
    target_energy = max(acc["target_energy"], 1e-30)
    forward_kl = max(acc["forward_full_kl"], 1e-30)
    reverse_kl = max(acc["reverse_full_kl"], 1e-30)
    routes = {}
    for name, row in acc["routes"].items():
        routes[name] = {
            "fisher_fidelity": 1 - row["error"] / target_energy,
            "fisher_cosine": row["cross"] / max(
                math.sqrt(row["prediction_energy"] * target_energy), 1e-30,
            ),
            "target_projection": row["cross"] / target_energy,
            "forward_kl_ratio": row["forward_kl"] / forward_kl,
            "reverse_error_kl_ratio": row["reverse_error_kl"] / reverse_kl,
        }
    shapley = {factor: acc["shapley_cross"][factor] / target_energy for factor in FACTORS}
    return {
        "examples": acc["count"],
        "target_fisher_energy": acc["target_energy"],
        "mean_forward_full_kl": acc["forward_full_kl"] / max(acc["count"], 1),
        "mean_reverse_full_kl": acc["reverse_full_kl"] / max(acc["count"], 1),
        "max_base_recomposition_error": acc["max_base_error"],
        "max_changed_recomposition_error": acc["max_changed_error"],
        "routes": routes,
        "shapley_target_projection": shapley,
        "shapley_absolute_sum": sum(abs(value) for value in shapley.values()),
        "shapley_sum": sum(shapley.values()),
        "median_example_shapley": {
            factor: float(np.median(acc["per_example"][factor])) for factor in FACTORS
        },
    }


@torch.inference_mode()
def evaluate(model, tokenizer, spec, config):
    rows = load_rows(spec)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) != 1:
        raise RuntimeError(f"newline is not one token: {newline}")
    accumulators = [empty_accumulator() for _ in model.blocks]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda")
        keep = base_ids[:, config["position"]] != newline[0]
        if not bool(keep.any()):
            continue
        base_ids = base_ids[keep]
        changed_ids = base_ids.clone()
        changed_ids[:, config["position"]] = newline[0]
        base_boundaries, native_base = trace(model, base_ids)
        changed_boundaries, native_changed = trace(model, changed_ids)
        for layer in range(len(model.blocks)):
            worlds = suffix_worlds(model, layer, base_boundaries[layer], changed_boundaries[layer])
            update_accumulator(accumulators[layer], worlds, native_base, native_changed)
    return {str(layer): finalize(acc) for layer, acc in enumerate(accumulators)}


def profile_correlation(results, factor, first, second):
    a = [results[first][str(layer)]["shapley_target_projection"][factor] for layer in range(12)]
    b = [results[second][str(layer)]["shapley_target_projection"][factor] for layer in range(12)]
    def ranks(values):
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result
    ar, br = ranks(a), ranks(b)
    if ar.std() == 0 or br.std() == 0:
        return 1.0 if np.array_equal(ar, br) else 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluating", corpus, flush=True)
        results[corpus] = evaluate(model, tokenizer, spec, config)
    correlations = {
        factor: {
            corpus: profile_correlation(results, factor, "fineweb", corpus)
            for corpus in ("wikitext", "tinystories")
        }
        for factor in FACTORS
    }
    all_rows = [results[corpus][str(layer)] for corpus in results for layer in range(12)]
    hypotheses = {
        "C1_exact_hybrid_recomposition": {"supported": all(
            max(row["max_base_recomposition_error"], row["max_changed_recomposition_error"]) <= 1e-4
            for row in all_rows
        )},
        "C2_nonresidual_carriers_are_material": {"supported": all(
            max(abs(results[c][str(layer)]["shapley_target_projection"]["embedding_stream"])
                + abs(results[c][str(layer)]["shapley_target_projection"]["value_bus"])
                for layer in range(11)) >= .20 for c in results
        )},
        "C3_embedding_stream_is_material": {"supported": all(
            max(abs(results[c][str(layer)]["shapley_target_projection"]["embedding_stream"])
                for layer in range(11)) >= .10 for c in results
        )},
        "C4_value_bus_is_material": {"supported": all(
            max(abs(results[c][str(layer)]["shapley_target_projection"]["value_bus"])
                for layer in range(11)) >= .10 for c in results
        )},
        "C5_late_residual_is_nearly_sufficient": {"supported": all(
            results[c]["10"]["routes"]["residual_only"]["fisher_fidelity"] >= .75 for c in results
        )},
        "C6_carrier_depth_profiles_transfer_ood": {"supported": all(
            value >= .70 for row in correlations.values() for value in row.values()
        ), "correlations": correlations},
        "C7_carrier_interaction_or_cancellation_is_material": {"supported": all(
            max(results[c][str(layer)]["shapley_absolute_sum"] for layer in range(11)) >= 1.20
            for c in results
        )},
        "C8_final_boundary_collapses_to_residual": {"supported": all(
            results[c]["11"]["routes"]["residual_only"]["fisher_fidelity"] >= .999
            and abs(results[c]["11"]["shapley_target_projection"]["embedding_stream"]) <= 1e-5
            and abs(results[c]["11"]["shapley_target_projection"]["value_bus"]) <= 1e-5
            for c in results
        )},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "profile_correlations": correlations, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    corpora = list(results)
    layers = np.arange(12)
    colors = dict(zip(FACTORS, ("#4c78a8", "#f58518", "#54a24b")))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for factor in FACTORS:
        values = np.asarray([[results[c][str(l)]["shapley_target_projection"][factor] for l in layers] for c in corpora])
        axes[0].plot(layers, values.mean(0), marker="o", color=colors[factor], label=factor)
        axes[0].fill_between(layers, values.min(0), values.max(0), color=colors[factor], alpha=.13)
    axes[0].axhline(0, color="black", lw=.8); axes[0].set_title("Signed carrier Shapley projection")
    for corpus in corpora:
        axes[1].plot(layers, [results[corpus][str(l)]["routes"]["residual_only"]["fisher_fidelity"] for l in layers], marker="o", label=corpus)
    axes[1].axhline(.75, color="black", ls="--", lw=.8); axes[1].set_title("Changed-residual-only fidelity")
    for corpus in corpora:
        axes[2].plot(layers, [results[corpus][str(l)]["shapley_absolute_sum"] for l in layers], marker="o", label=corpus)
    axes[2].axhline(1, color="black", lw=.8); axes[2].axhline(1.2, color="black", ls="--", lw=.8); axes[2].set_title("Absolute Shapley sum")
    for axis in axes:
        axis.set_xlabel("boundary after layer")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Full-token carrier decomposition across depth",
        "",
        "An ordinary final token is replaced by a newline. At every layer boundary, all eight base/changed combinations of the complete residual sequence, the original normalized embedding stream, and the persistent Layer-0 attention value bus are propagated through the untouched model suffix. Exact vector Shapley effects are projected onto the complete changed-logit direction in the baseline Fisher metric.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Carrier depth audit](bilin12_boundary_carrier_depth_audit.png)",
        "",
        "| Corpus | Boundary | Residual fidelity | Residual / embed / bus Shapley | Abs. sum | Residual target-error KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        for layer in range(12):
            row = results[corpus][str(layer)]
            shapley = row["shapley_target_projection"]
            lines.append(
                f"| {corpus} | {layer} | {row['routes']['residual_only']['fisher_fidelity']:.3f} | "
                f"{shapley['residual_sequence']:.3f} / {shapley['embedding_stream']:.3f} / {shapley['value_bus']:.3f} | "
                f"{row['shapley_absolute_sum']:.3f} | {row['routes']['residual_only']['reverse_error_kl_ratio']:.3f} |"
            )
    lines += [
        "",
        "## OOD depth-profile correlations",
        "",
        "| Carrier | FineWeb↔WikiText | FineWeb↔TinyStories |",
        "|---|---:|---:|",
    ]
    for factor in FACTORS:
        lines.append(f"| {factor} | {correlations[factor]['wikitext']:.3f} | {correlations[factor]['tinystories']:.3f} |")
    lines += [
        "",
        "A boundary carrier is an interface, not a storage location. Signed Shapley projection answers how a base/changed hybrid participates in this particular finite edit under a declared suffix and metric. Values above one or below zero reveal cancellation or interaction; they are not percentages of information. The final boundary is a deliberate sanity check: after all blocks have run, embedding and value-bus effects have already been compiled into the residual.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
