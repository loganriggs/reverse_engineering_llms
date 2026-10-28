#!/usr/bin/env python3
"""Discover source variables from realized routing and local operator effects."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_model_derived_source_ontology import spherical_kmeans, weighted_nmi
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports
from run_bilin18_typed_attention_edge_vocabulary import (
    CATEGORIES,
    distance_ids,
    rmsnorm_pullback_readers,
    token_categories,
)


CONFIG = Path("configs/bilin18_causal_source_ontology.json")
OUTPUT = TENSOR_ROOT / "bilin18_causal_source_ontology.json"
ASSIGNMENTS = TENSOR_ROOT / "bilin18_causal_source_ontology_assignments.npz"
PROFILES = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
REPORT = TENSOR_ROOT / "BILIN18_CAUSAL_SOURCE_ONTOLOGY.md"
PARTIAL = TENSOR_ROOT / "bilin18_causal_source_ontology.partial.json"


def normalize_profiles(sums, counts):
    means = sums / counts.clamp_min(1)
    global_mean = sums.sum(0) / counts.sum(0).clamp_min(1)
    relative = torch.log1p(means / global_mean.clamp_min(1e-12))
    return F.normalize(relative, dim=1)


@torch.inference_mode()
def collect_source_profiles(model, raw, vocabulary_size, config, seed_offset):
    heads = model.config.n_head
    dimension = model.config.n_embd
    distances = len(config["distance_bins"]) - 1
    per_layer = heads * distances
    coordinates = len(config["layers"]) * per_layer
    routing = torch.zeros(vocabulary_size, coordinates, device="cuda")
    operator = torch.zeros_like(routing)
    counts = torch.zeros_like(routing)
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    head_id = torch.arange(heads, device="cuda")[None, None, :, None]
    local_coordinate = head_id * distances + distance[None, :, None, :]
    valid_base = distance[None, :, None, :] >= 0
    probes = {
        layer: operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + seed_offset + layer)[0]
        for layer in config["layers"]
    }
    flat_routing = routing.view(-1); flat_operator = operator.view(-1); flat_counts = counts.view(-1)
    for start in range(0, len(raw), config["profile_batch_size"]):
        if start % 64 == 0:
            print("profile rows", start, "/", len(raw), flush=True)
        ids = torch.as_tensor(raw[start:start + config["profile_batch_size"], :config["sequence_length"]], device="cuda")
        batch = len(ids)
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        token_grid = ids[:, None, None, :].expand(batch, len(config["positions"]), heads, config["sequence_length"])
        for layer, block in enumerate(model.blocks[:max(config["layers"]) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in config["layers"]:
                pre = mixed[:, config["positions"]].float() + details["write"][:, config["positions"]].float()
                readers = rmsnorm_pullback_readers(pre, probes[layer])
                projection = block.attn.c_proj.weight.float().T.reshape(heads, dimension // heads, dimension)
                head_readers = torch.einsum("bpqo,hdo->bphqd", readers, projection)
                value_effect = torch.einsum("bkhd,bphqd->bphqk", details["mixed_value"].float(), head_readers)
                pattern = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3).float()
                operator_magnitude = (value_effect * pattern[:, :, :, None]).square().sum(3).sqrt()
                routing_magnitude = pattern.abs()
                coordinate = (layer * per_layer + local_coordinate).expand(batch, -1, -1, -1)
                valid = valid_base.expand(batch, -1, heads, -1)
                global_index = token_grid[valid] * coordinates + coordinate[valid]
                flat_routing.scatter_add_(0, global_index, routing_magnitude[valid])
                flat_operator.scatter_add_(0, global_index, operator_magnitude[valid])
                flat_counts.scatter_add_(0, global_index, torch.ones_like(routing_magnitude[valid]))
            pre_full = mixed + details["write"]
            state = F.rms_norm(pre_full, (dimension,))
            x = pre_full + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return {
        "routing": normalize_profiles(routing, counts).cpu(),
        "operator": normalize_profiles(operator, counts).cpu(),
        "counts": counts.sum(1).cpu(),
    }


def descriptive_clusters(assignments, human, frequencies, tokenizer, count):
    rows = []
    total = max(frequencies.sum(), 1e-30)
    for cluster in range(count):
        members = np.flatnonzero(assignments == cluster)
        mass = np.bincount(human[members], weights=frequencies[members], minlength=count)
        probabilities = mass / max(mass.sum(), 1e-30)
        nonzero = probabilities > 0
        entropy = float(-(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum())
        ordered = members[np.argsort(frequencies[members])[::-1]][:16]
        rows.append({
            "cluster": cluster,
            "frequency_mass": float(frequencies[members].sum() / total),
            "human_purity": float(probabilities.max()),
            "human_entropy_bits": entropy,
            "human_plurality": CATEGORIES[int(mass.argmax())],
            "top_tokens": [tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t") for index in ordered],
        })
    return rows


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    profile_a_raw = np.asarray(corpus[config["profile_a_start"]:config["profile_a_start"] + config["profile_rows"], :config["sequence_length"]], dtype=np.int64)
    profile_b_raw = np.asarray(corpus[config["profile_b_start"]:config["profile_b_start"] + config["profile_rows"], :config["sequence_length"]], dtype=np.int64)
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    frequency_raw = np.concatenate((profile_a_raw.reshape(-1), profile_b_raw.reshape(-1)))
    frequencies = np.bincount(frequency_raw, minlength=tokenizer.vocab_size).astype(np.float64)
    weights = np.power(frequencies + 1e-3, config["frequency_power"])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("collect causal source profile A", flush=True)
    profile_a = collect_source_profiles(model, profile_a_raw, tokenizer.vocab_size, config, 0)
    print("collect causal source profile B", flush=True)
    profile_b = collect_source_profiles(model, profile_b_raw, tokenizer.vocab_size, config, 100)
    torch.save({"a": profile_a, "b": profile_b}, PROFILES)
    embedding = F.normalize(model.embedding.weight[:tokenizer.vocab_size].detach().float().cpu(), dim=1)
    weight_tensor = torch.as_tensor(weights, device="cuda")
    feature_sets = {
        "embedding_exact": embedding,
        "routing_profile_a": profile_a["routing"],
        "operator_profile_a": profile_a["operator"],
        "operator_profile_b": profile_b["operator"],
        "hybrid_operator_embedding_a": F.normalize(torch.cat((embedding, profile_a["operator"]), 1), dim=1),
    }
    assignments = {"human": human}; distortions = {}
    for index, (name, features) in enumerate(feature_sets.items()):
        print("cluster causal ontology", name, flush=True)
        cluster, _, distortion = spherical_kmeans(
            features.to("cuda"), weight_tensor, config["category_count"],
            config["kmeans_iterations"], config["kmeans_restarts"], config["seed"] + 1000 * index,
        )
        assignments[name] = cluster.cpu().numpy().astype(np.int64); distortions[name] = distortion
    np.savez_compressed(ASSIGNMENTS, **assignments)
    results = {}; maximum_error = 0.0
    saved = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    if saved.get("config") == config:
        results = saved.get("results", {}); maximum_error = float(saved.get("maximum_error", 0.0))
    for name in config["ontologies"]:
        if name in results:
            print("evaluate causal ontology", name, "(resumed)", flush=True); continue
        print("evaluate causal ontology", name, flush=True)
        supports = select_supports(model, discovery, assignments[name], config)
        row, error = evaluate(model, evaluation, assignments[name], config, supports)
        results[name] = row["5"]; maximum_error = max(maximum_error, error)
        PARTIAL.write_text(json.dumps({"config": config, "maximum_error": maximum_error, "results": results}, indent=2))
    score = {name: results[name]["methods"]["fine128"]["kl_recovery"] for name in config["ontologies"]}
    top1 = {name: results[name]["methods"]["fine128"]["top1_agreement"] for name in config["ontologies"]}
    stability = weighted_nmi(assignments["operator_profile_a"], assignments["operator_profile_b"], frequencies, config["category_count"])
    hypotheses = {
        "P1_operator_profile_beats_embedding": {"supported": score["operator_profile_a"] >= score["embedding_exact"] + .05, "difference": score["operator_profile_a"] - score["embedding_exact"]},
        "P2_operator_profile_is_competitive_with_human": {"supported": score["operator_profile_a"] >= score["human"] - .03, "difference": score["operator_profile_a"] - score["human"]},
        "P3_operator_profiles_are_split_stable": {"supported": stability >= .60, "weighted_nmi": stability},
        "P4_operator_profile_behavior_is_split_stable": {"supported": abs(score["operator_profile_a"] - score["operator_profile_b"]) <= .05, "absolute_difference": abs(score["operator_profile_a"] - score["operator_profile_b"])},
        "P5_hybrid_improves_operator_profile": {"supported": score["hybrid_operator_embedding_a"] >= score["operator_profile_a"] + .03, "difference": score["hybrid_operator_embedding_a"] - score["operator_profile_a"]},
        "P6_a_causal_learned_ontology_is_prediction_useful": {"supported": max(top1[name] for name in config["ontologies"] if name != "human") >= .70},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    summaries = {name: descriptive_clusters(assignments[name], human, frequencies, tokenizer, config["category_count"]) for name in config["ontologies"] if name != "human"}
    nmi_human = {name: weighted_nmi(assignments[name], human, frequencies, config["category_count"]) for name in config["ontologies"]}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "distortions": distortions, "weighted_nmi_with_human": nmi_human, "cluster_summaries": summaries, "results": results}, indent=2, allow_nan=False))
    PARTIAL.unlink(missing_ok=True)
    lines = [
        "# Causal source-variable ontology", "",
        "Static weight geometry failed to recover the human typed-edge program. This red team instead clusters each token by its realized Layer-0--5 profile across heads and distance bins: absolute QK routing magnitude or weight-derived effect on the receiving normalized bilinear operator. All partitions retain 21 categories and 128 selected types per layer.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Ontology | KL recovery | CE delta | Top-1 | NMI with human |", "|---|---:|---:|---:|---:|"]
    for name in config["ontologies"]:
        value = results[name]["methods"]["fine128"]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} | {nmi_human[name]:.3f} |")
    candidates = sorted(summaries["operator_profile_a"], key=lambda row: (row["human_purity"], -row["frequency_mass"]))
    lines += ["", "## Candidate variables not captured cleanly by the human vocabulary", ""]
    for row in candidates[:10]:
        tokens = ", ".join(repr(token) for token in row["top_tokens"][:10])
        lines.append(f"- Cluster {row['cluster']}: mass {row['frequency_mass']:.1%}, human purity {row['human_purity']:.2f}, entropy {row['human_entropy_bits']:.2f} bits, plurality `{row['human_plurality']}`; {tokens}")
    lines += [
        "", "## Guardrails", "",
        "These profiles are learned from realized FineWeb states and therefore describe corpus-conditioned source roles, not a weight-only universal ontology. Magnitudes discard the sign and detailed operator-probe vector, so distinct functions can still collide. Token identity remains static after clustering: the same token cannot change category by context. Low human purity nominates an unknown variable but does not validate it; split stability, causal program performance, and targeted examples must agree before assigning meaning.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "scores": score, "top1": top1, "stability": stability}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
