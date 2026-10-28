#!/usr/bin/env python3
"""Search for reusable bus-aware value/write operator templates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_model_derived_source_ontology import spherical_kmeans, weighted_nmi
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import factored_frobenius_cosines, leading_mode, spearman


CONFIG = Path("configs/bilin18_bus_aware_function_templates.json")
OUTPUT = TENSOR_ROOT / "bilin18_bus_aware_function_templates.json"
TENSORS = TENSOR_ROOT / "bilin18_bus_aware_function_templates.pt"
REPORT = TENSOR_ROOT / "BILIN18_BUS_AWARE_FUNCTION_TEMPLATES.md"


def rank_summary(gram):
    eigenvalues = np.linalg.eigvalsh(gram).clip(0)[::-1]
    cumulative = np.cumsum(eigenvalues) / max(eigenvalues.sum(), 1e-30)
    return {
        "rank90": int(np.searchsorted(cumulative, .90) + 1),
        "rank50": int(np.searchsorted(cumulative, .50) + 1),
        "participation_ratio": float(eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-30)),
        "top16_energy": float(cumulative[min(15, len(cumulative) - 1)]),
        "eigenvalues": eigenvalues.tolist(),
    }


def normalized_gram(signatures):
    signatures = F.normalize(signatures.float(), dim=1)
    return (signatures @ signatures.T).cpu().numpy(), signatures


@torch.inference_mode()
def natural_signatures(model, raw, positions, projection, config):
    heads = model.config.n_head; dimension = model.config.n_embd; head_dim = dimension // heads
    chunks = [[] for _ in range(len(config["layers"]) * heads)]
    for start in range(0, len(raw), config["profile_batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["profile_batch_size"], :config["sequence_length"]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks[:max(config["layers"]) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in config["layers"]:
                values = details["mixed_value"][:, positions].float()
                for head in range(heads):
                    sl = slice(head * head_dim, (head + 1) * head_dim)
                    output = block.attn.c_proj.weight[:, sl].float()
                    reduced = output.T @ projection
                    response = values[:, :, head] @ reduced
                    chunks[layer * heads + head].append(response.cpu().flatten())
            pre = mixed + details["write"]
            state = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return torch.stack([torch.cat(rows) for rows in chunks])


def main():
    config = json.loads(CONFIG.read_text())
    corpus = np.load(config["corpus"], mmap_mode="r")
    raw_a = np.asarray(corpus[config["profile_a_start"]:config["profile_a_start"] + config["profile_rows"], :config["sequence_length"]], dtype=np.int64)
    raw_b = np.asarray(corpus[config["profile_b_start"]:config["profile_b_start"] + config["profile_rows"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    heads = model.config.n_head; dimension = model.config.n_embd; head_dim = dimension // heads
    bus_factors = []; current_factors = []; output_modes = []; bus_fractions = []
    labels = []
    for layer in config["layers"]:
        block = model.blocks[layer].attn
        lamb = float(block.lamb.detach())
        for head in range(heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            output = block.c_proj.weight[:, sl].detach().float().cpu()
            value = block.c_v.weight[sl].detach().float().cpu()
            selector = torch.zeros(head_dim, dimension)
            selector[:, sl] = torch.eye(head_dim)
            combined = torch.cat(((1 - lamb) * value, lamb * selector), dim=1)
            bus_factors.append((output, combined)); current_factors.append((output, value))
            output_mode, _, _ = leading_mode(output, combined)
            output_modes.append(output_mode)
            current_norm = torch.sum((output.T @ output) * (value @ value.T).T) * (1 - lamb) ** 2
            bus_norm = torch.sum(output.T @ output) * lamb ** 2
            bus_fractions.append(float(bus_norm / (current_norm + bus_norm).clamp_min(1e-30)))
            labels.append(f"L{layer}H{head}")
    print("bus-aware weight geometry", flush=True)
    bus_cosine, bus_norms = factored_frobenius_cosines(bus_factors)
    current_cosine, _ = factored_frobenius_cosines(current_factors)
    bus_gram = bus_cosine * (bus_norms[:, None] * bus_norms[None, :])
    weight_rank = rank_summary(bus_gram)
    triangle = np.triu_indices(len(labels), 1)
    geometry_correlation = spearman(current_cosine[triangle], bus_cosine[triangle])
    mode_matrix = F.normalize(torch.stack(output_modes), dim=1)
    mode_cosine = (mode_matrix @ mode_matrix.T).numpy()
    nearest = []
    nearest_pairs = []
    for index, label in enumerate(labels):
        values = np.abs(mode_cosine[index]).copy(); values[index] = -1
        neighbor = int(values.argmax()); nearest.append(float(values[neighbor]))
        nearest_pairs.append({"function": label, "neighbor": labels[neighbor], "abs_cosine": float(values[neighbor])})
    rng = np.random.default_rng(config["seed"])
    null_medians = []
    for _ in range(256):
        random = rng.normal(size=(len(labels), dimension)); random /= np.linalg.norm(random, axis=1, keepdims=True)
        cosine = np.abs(random @ random.T); np.fill_diagonal(cosine, -1)
        null_medians.append(float(np.median(cosine.max(1))))
    random_p95 = float(np.quantile(null_medians, .95)); observed_nearest = float(np.median(nearest))
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    projection = torch.randn(dimension, config["output_sketch"], generator=generator, device="cuda")
    projection = torch.linalg.qr(projection, mode="reduced").Q
    positions = torch.as_tensor(config["positions"], device="cuda")
    print("natural signature A", flush=True)
    signatures_a = natural_signatures(model, raw_a, positions, projection, config)
    print("natural signature B", flush=True)
    signatures_b = natural_signatures(model, raw_b, positions, projection, config)
    natural_a, normalized_a = normalized_gram(signatures_a)
    natural_b, normalized_b = normalized_gram(signatures_b)
    natural_stability = float(np.corrcoef(natural_a[triangle], natural_b[triangle])[0, 1])
    equal = torch.ones(len(labels), device="cuda")
    cluster_a, _, _ = spherical_kmeans(normalized_a.to("cuda"), equal, config["cluster_count"], config["kmeans_iterations"], config["kmeans_restarts"], config["seed"] + 100)
    cluster_b, _, _ = spherical_kmeans(normalized_b.to("cuda"), equal, config["cluster_count"], config["kmeans_iterations"], config["kmeans_restarts"], config["seed"] + 200)
    cluster_a = cluster_a.cpu().numpy(); cluster_b = cluster_b.cpu().numpy()
    cluster_nmi = weighted_nmi(cluster_a, cluster_b, np.ones(len(labels)), config["cluster_count"])
    natural_rank_a = rank_summary(natural_a); natural_rank_b = rank_summary(natural_b)
    clusters = []
    for cluster in range(config["cluster_count"]):
        members = [labels[index] for index in np.flatnonzero(cluster_a == cluster)]
        clusters.append({"cluster": cluster, "members_a": members, "size": len(members)})
    hypotheses = {
        "T1_full_bus_aware_operators_have_small_template_bank": {"supported": weight_rank["rank90"] <= 16, "rank90": weight_rank["rank90"], "top16_energy": weight_rank["top16_energy"]},
        "T2_output_verbs_recur_beyond_random_geometry": {"supported": observed_nearest >= random_p95 + .05, "observed_median_nearest": observed_nearest, "random_median_nearest_p95": random_p95, "difference": observed_nearest - random_p95},
        "T3_natural_function_geometry_is_split_stable": {"supported": natural_stability >= .80, "correlation": natural_stability},
        "T4_natural_function_clusters_are_stable": {"supported": cluster_nmi >= .60, "nmi": cluster_nmi},
        "T5_value_bus_materially_changes_function_geometry": {"supported": geometry_correlation < .90, "current_bus_similarity_spearman": geometry_correlation},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {
        "config": config, "hypotheses": hypotheses, "weight_rank": weight_rank,
        "natural_rank_a": natural_rank_a, "natural_rank_b": natural_rank_b,
        "bus_fraction": {label: value for label, value in zip(labels, bus_fractions)},
        "nearest_output_verbs": sorted(nearest_pairs, key=lambda row: row["abs_cosine"], reverse=True),
        "clusters_a": clusters,
        "cluster_assignments_a": dict(zip(labels, cluster_a.tolist())),
        "cluster_assignments_b": dict(zip(labels, cluster_b.tolist())),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save({"labels": labels, "bus_cosine": torch.from_numpy(bus_cosine), "current_cosine": torch.from_numpy(current_cosine), "natural_a": torch.from_numpy(natural_a), "natural_b": torch.from_numpy(natural_b), "output_modes": mode_matrix, "cluster_a": torch.from_numpy(cluster_a), "cluster_b": torch.from_numpy(cluster_b)}, TENSORS)
    lines = [
        "# Bus-aware reusable function templates", "",
        "Each Layer-0--5 head is represented by its exact value-side map from two arguments—the current normalized residual and the persistent Layer-0 value bus—to the residual write. This corrects the current-value-only OV atlas. Full operator PCA, leading output-verb recurrence, and natural-response clustering test three different meanings of a reusable function template.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "## Summary", "",
        f"- Full weight-operator rank90: `{weight_rank['rank90']}` of {len(labels)}; top-16 energy `{weight_rank['top16_energy']:.3f}`; participation ratio `{weight_rank['participation_ratio']:.1f}`.",
        f"- Median bus-input Frobenius fraction: `{np.median(bus_fractions):.3f}`.",
        f"- Current-only versus bus-aware similarity Spearman: `{geometry_correlation:.3f}`.",
        f"- Leading output-verb median nearest cosine: `{observed_nearest:.3f}` versus random p95 `{random_p95:.3f}`.",
        f"- Natural-response geometry A/B correlation: `{natural_stability:.3f}`; cluster NMI `{cluster_nmi:.3f}`.",
        f"- Natural-response rank90 A/B: `{natural_rank_a['rank90']}/{natural_rank_b['rank90']}`.",
        "", "Strongest leading-output neighbors:", "",
    ]
    for row in sorted(nearest_pairs, key=lambda item: item["abs_cosine"], reverse=True)[:12]:
        lines.append(f"- {row['function']} ↔ {row['neighbor']}: `{row['abs_cosine']:.3f}`")
    lines += ["", "Discovery-A natural-response clusters:", ""]
    for row in clusters: lines.append(f"- C{row['cluster']}: {', '.join(row['members_a'])}")
    lines += [
        "", "## Guardrails", "",
        "The full bus-aware map is exact only for the value/write branch; QK routing, RMSNorm, and downstream computation remain outside it. Linear template rank, leading output directions, and clusters are distinct ontologies and need not agree. Natural signatures use a frozen random output sketch and aligned token positions but different texts across splits. Stable clustering would nominate function families, not prove semantic readability or causal interchangeability. Any surviving family requires a fresh finite replacement test.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "weight_rank": weight_rank, "natural_rank_a": natural_rank_a}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
