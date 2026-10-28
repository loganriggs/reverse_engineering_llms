#!/usr/bin/env python3
"""Compare human source types with equal-budget model-derived token ontologies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG = Path("configs/bilin18_model_derived_source_ontology.json")
OUTPUT = TENSOR_ROOT / "bilin18_model_derived_source_ontology.json"
ASSIGNMENTS = TENSOR_ROOT / "bilin18_model_derived_source_ontology_assignments.npz"
REPORT = TENSOR_ROOT / "BILIN18_MODEL_DERIVED_SOURCE_ONTOLOGY.md"
PARTIAL = TENSOR_ROOT / "bilin18_model_derived_source_ontology.partial.json"


def spherical_kmeans(features, weights, clusters, iterations, restarts, seed):
    """Weighted spherical k-means; returns assignments and weighted distortion."""
    features = F.normalize(features.float(), dim=1)
    weights = weights.float().clamp_min(0)
    probabilities = weights / weights.sum().clamp_min(1e-30)
    best = None
    for restart in range(restarts):
        generator = torch.Generator(device=features.device).manual_seed(seed + restart)
        initial = torch.multinomial(probabilities, clusters, replacement=False, generator=generator)
        centers = features[initial]
        assignments = None
        for _ in range(iterations):
            assignments = (features @ centers.T).argmax(1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assignments, features * weights[:, None])
            mass = torch.zeros(clusters, device=features.device)
            mass.index_add_(0, assignments, weights)
            empty = mass == 0
            if empty.any():
                replacement = torch.multinomial(probabilities, int(empty.sum()), replacement=False, generator=generator)
                sums[empty] = features[replacement]
            centers = F.normalize(sums, dim=1)
        similarity = (features * centers[assignments]).sum(1)
        distortion = float((weights * (1 - similarity)).sum() / weights.sum().clamp_min(1e-30))
        if best is None or distortion < best[2]:
            best = (assignments.clone(), centers.clone(), distortion)
    return best


@torch.inference_mode()
def random_projection(input_dimension, output_dimension, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(input_dimension, output_dimension, generator=generator, device=device) / math.sqrt(output_dimension)


@torch.inference_mode()
def model_signatures(model, config, vocabulary_size=None):
    """Static token signatures from weights evaluated on the repeated token embedding."""
    device = "cuda"
    dimension = model.config.n_embd
    vocab = model.config.vocab_size if vocabulary_size is None else vocabulary_size
    batch_size = 2048
    embedding_projection = random_projection(dimension, config["signature_dimension"], config["seed"] + 1, device)
    value_projections = [
        random_projection(dimension, config["signature_dimension"] // len(config["layers"]), config["seed"] + 10 + layer, device)
        for layer in config["layers"]
    ]
    qkv_projections = {
        (layer, role): random_projection(dimension, config["qkv_component_dimension"], config["seed"] + 100 + 10 * layer + role_index, device)
        for layer in config["layers"]
        for role_index, role in enumerate(("k", "k2", "v"))
    }
    outputs = {"embedding": [], "value": [], "qkv": []}
    for start in range(0, vocab, batch_size):
        stop = min(start + batch_size, vocab)
        token_ids = torch.arange(start, stop, device=device)
        x0 = F.rms_norm(model.embedding(token_ids).float(), (dimension,))
        outputs["embedding"].append((x0 @ embedding_projection).cpu())
        values, qkv = [], []
        for layer in config["layers"]:
            block = model.blocks[layer]
            normalized = F.rms_norm(x0, (dimension,))
            value = block.attn.c_v(normalized).float()
            values.append(F.normalize(value, dim=1) @ value_projections[layer])
            for role, module in (("k", block.attn.c_k), ("k2", block.attn.c_k2), ("v", block.attn.c_v)):
                role_value = module(normalized).float()
                qkv.append(F.normalize(role_value, dim=1) @ qkv_projections[(layer, role)])
        outputs["value"].append(torch.cat(values, 1).cpu())
        outputs["qkv"].append(torch.cat(qkv, 1).cpu())
    return {name: F.normalize(torch.cat(chunks).float(), dim=1) for name, chunks in outputs.items()}


def weighted_nmi(first, second, weights, categories):
    table = np.zeros((categories, categories), dtype=np.float64)
    np.add.at(table, (first, second), weights)
    table /= max(table.sum(), 1e-30)
    row = table.sum(1); column = table.sum(0)
    valid = table > 0
    expected = row[:, None] * column[None, :]
    mutual = float((table[valid] * np.log(table[valid] / expected[valid])).sum())
    entropy_first = float(-(row[row > 0] * np.log(row[row > 0])).sum())
    entropy_second = float(-(column[column > 0] * np.log(column[column > 0])).sum())
    return mutual / max(math.sqrt(entropy_first * entropy_second), 1e-30)


def cluster_summaries(assignments, human, frequencies, tokenizer, count):
    summaries = []
    for cluster in range(count):
        members = np.flatnonzero(assignments == cluster)
        ordered = members[np.argsort(frequencies[members])[::-1]][:12]
        human_mass = np.bincount(human[members], weights=frequencies[members], minlength=count)
        summaries.append({
            "cluster": cluster,
            "frequency_mass": float(frequencies[members].sum() / max(frequencies.sum(), 1e-30)),
            "top_human_category": CATEGORIES[int(human_mass.argmax())],
            "top_tokens": [tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t") for index in ordered],
        })
    return summaries


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    human = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    ontology = np.asarray(
        corpus[config["ontology_start"]:config["ontology_start"] + config["ontology_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    discovery = np.asarray(
        corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1],
        dtype=np.int64,
    )
    frequencies = np.bincount(ontology.reshape(-1), minlength=tokenizer.vocab_size).astype(np.float64)
    weights = np.power(frequencies + 1e-3, config["frequency_power"])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    signatures = model_signatures(model, config, tokenizer.vocab_size)
    assignments = {"human": human}
    distortions = {}
    weight_tensor = torch.as_tensor(weights, device="cuda")
    for index, name in enumerate(("embedding", "value", "qkv")):
        print("fit source ontology", name, flush=True)
        cluster, _, distortion = spherical_kmeans(
            signatures[name].to("cuda"), weight_tensor, config["category_count"],
            config["kmeans_iterations"], config["kmeans_restarts"], config["seed"] + 1000 * index,
        )
        assignments[name] = cluster.cpu().numpy().astype(np.int64)
        distortions[name] = distortion
    generator = np.random.default_rng(config["seed"] + 9000)
    assignments["random"] = generator.integers(0, config["category_count"], size=tokenizer.vocab_size, dtype=np.int64)
    np.savez_compressed(ASSIGNMENTS, **assignments)
    results = {}
    maximum_error = 0.0
    saved = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    if saved.get("config") == config:
        results = saved.get("results", {})
        maximum_error = float(saved.get("maximum_error", 0.0))
    for name in config["ontologies"]:
        if name in results:
            print("evaluate source ontology", name, "(resumed)", flush=True)
            continue
        print("evaluate source ontology", name, flush=True)
        supports = select_supports(model, discovery, assignments[name], config)
        row, error = evaluate(model, evaluation, assignments[name], config, supports)
        results[name] = row["5"]
        maximum_error = max(maximum_error, error)
        PARTIAL.write_text(json.dumps({"config": config, "maximum_error": maximum_error, "results": results}, indent=2))
    score = {name: results[name]["methods"]["fine128"]["kl_recovery"] for name in config["ontologies"]}
    top1 = {name: results[name]["methods"]["fine128"]["top1_agreement"] for name in config["ontologies"]}
    learned = ("embedding", "value", "qkv")
    hypotheses = {
        "O1_qkv_is_competitive_with_human": {"supported": score["qkv"] >= score["human"] - .03, "difference": score["qkv"] - score["human"]},
        "O2_qkv_beats_embedding_only": {"supported": score["qkv"] >= score["embedding"] + .05, "difference": score["qkv"] - score["embedding"]},
        "O3_qkv_beats_random": {"supported": score["qkv"] >= score["random"] + .20, "difference": score["qkv"] - score["random"]},
        "O4_a_learned_ontology_improves_human": {"supported": max(score[name] for name in learned) >= score["human"] + .03, "best_difference": max(score[name] for name in learned) - score["human"]},
        "O5_a_learned_ontology_is_prediction_faithful": {"supported": max(top1[name] for name in learned) >= .75},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    nmi = {
        name: weighted_nmi(assignments[name], human, frequencies, config["category_count"])
        for name in config["ontologies"]
    }
    summaries = {name: cluster_summaries(assignments[name], human, frequencies, tokenizer, config["category_count"]) for name in learned}
    payload = {"config": config, "hypotheses": hypotheses, "distortions": distortions, "weighted_nmi_with_human": nmi, "cluster_summaries": summaries, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    PARTIAL.unlink(missing_ok=True)
    lines = [
        "# Model-derived source ontologies", "",
        "Five equal-budget source partitions are compared in the six-layer closed-loop typed-edge program: human token categories, spherical clusters of token embeddings, clusters of Layer-0--5 value signatures, clusters of Layer-0--5 key/key2/value signatures, and random token categories. Every ontology has 21 categories and receives 128 selected types per layer.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Ontology | KL recovery | CE delta | Top-1 agreement | Weighted NMI with human |", "|---|---:|---:|---:|---:|"]
    for name in config["ontologies"]:
        value = results[name]["methods"]["fine128"]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} | {nmi[name]:.3f} |")
    lines += ["", "## Representative learned clusters", ""]
    for name in learned:
        lines.append(f"### {name}")
        lines.append("")
        for cluster in summaries[name][:8]:
            tokens = ", ".join(repr(token) for token in cluster["top_tokens"][:8])
            lines.append(f"- Cluster {cluster['cluster']} ({cluster['frequency_mass']:.1%}; human plurality `{cluster['top_human_category']}`): {tokens}")
        lines.append("")
    lines += [
        "## Guardrails", "",
        "The learned categories are static token-identity partitions, not contextual source-state clusters. They use only model weights plus token frequencies from a disjoint ontology-fit corpus. Random projections make clustering tractable and preserve geometry only approximately. Spherical k-means optimizes within-cluster cosine distortion, not causal program fidelity; a negative result would not rule out learned causal categories. Human categories were engineered with linguistic knowledge, whereas learned clusters are unlabeled and may require a separate interpretability step.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "scores": score, "top1": top1, "nmi": nmi}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
