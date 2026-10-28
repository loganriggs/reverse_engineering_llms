#!/usr/bin/env python3
"""Compare frozen weight motifs with held-out supervised causal fingerprints.

The 50-label census is not treated as ground truth for the model's ontology.
Only its 32 competence-gated labels are used as external semantic anchors.  No
label enters weight-motif discovery.  Each attention head receives a signed
local-causal fingerprint: the derivative of a held-out feature logit contrast
with respect to a multiplicative head-write gate.  A shared vocabulary
permutation preserves the full feature-objective Gram matrix and serves as the
non-semantic control.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_supervised_anchor_motif_comparison.json")
OUTPUT = TENSOR_ROOT / "bilin18_supervised_anchor_motif_comparison.json"
ARRAYS = TENSOR_ROOT / "bilin18_supervised_anchor_motif_comparison.npz"
REPORT = TENSOR_ROOT / "BILIN18_SUPERVISED_ANCHOR_MOTIF_COMPARISON.md"
FIGURE = TENSOR_ROOT / "bilin18_supervised_anchor_motif_comparison.png"


def gated_forward(model, token_ids, positions, gates):
    """Exact checkpoint forward with one differentiable scale per head/example."""
    batch, length = token_ids.shape
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    x = F.rms_norm(model.embedding(token_ids), (dimension,))
    x0 = x
    value_bus = None
    causal = torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (dimension,))
        attention = block.attn

        def projected(module):
            value = module(normalized).view(batch, length, heads, head_dim)
            value = F.rms_norm(value, (head_dim,))
            cosine, sine = attention.rotary(value)
            return model.tt.apply_rotary_emb(value, cosine, sine)

        q, k = projected(attention.c_q), projected(attention.c_k)
        q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
        current_value = attention.c_v(normalized).view(batch, length, heads, head_dim)
        first_value = current_value if value_bus is None else value_bus.view_as(current_value)
        if value_bus is None:
            value_bus = first_value
        mixed_value = (1 - attention.lamb) * current_value + attention.lamb * first_value
        score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_dim
        score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_dim
        pattern = (score1 * score2).masked_fill(~causal, 0)
        attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value)
        head_values = attended.transpose(1, 2)
        # Preserve the checkpoint's native all-head contraction bit-for-bit at
        # gate=1, then add only the gated per-head deltas.  Multiplying the
        # concatenated head tensor by ones changes bfloat16 contraction layout
        # enough to accumulate visible logit drift over 18 layers.
        native_write = attention.c_proj(head_values.contiguous().view(batch, length, dimension))
        projection = attention.c_proj.weight.view(dimension, heads, head_dim)
        head_writes = torch.einsum("bqhd,ohd->bqho", head_values, projection)
        delta = (gates[:, layer] - 1)[:, None, :, None].to(head_writes.dtype)
        write = native_write + (delta * head_writes).sum(2)
        pre = mixed + write
        mlp_input = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(mlp_input) * block.mlp.Right(mlp_input)
        # Preserve checkpoint association: the MLP forms Down(product)+bias
        # internally before that complete write is added to the residual.
        mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + mlp_write
    selected = F.rms_norm(x, (dimension,))[torch.arange(batch, device=x.device), positions]
    raw = model.lm_head(selected).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def native_forward_positions(model, token_ids, positions):
    """Checkpoint-native forward, retaining the same sequence/batch shape."""
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(token_ids), (dimension,))
    x0 = x
    value_bus = None
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
    selected = F.rms_norm(x, (dimension,))[torch.arange(len(token_ids), device=x.device), positions]
    return 30 * torch.tanh(model.lm_head(selected).float() / 30)


def unit_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)


def cosine_rows(values):
    normalized = unit_rows(values)
    return normalized @ normalized.T


def pearson_triangle(first, second):
    triangle = np.triu_indices(len(first), 1)
    return float(np.corrcoef(first[triangle], second[triangle])[0, 1])


def components_and_pairs(payload, heads):
    components = []
    for component in payload["bus_robust"]["components"]:
        components.append([member["layer"] * heads + member["head"] for member in component])
    membership = {node: component_index for component_index, component in enumerate(components)
                  for node in component}
    closure_pairs = []
    for component_index, component in enumerate(components):
        for left_index in range(len(component)):
            for right_index in range(left_index + 1, len(component)):
                closure_pairs.append((component[left_index], component[right_index], component_index))
    direct_pairs = [(int(left), int(right), membership[int(left)])
                    for left, right in payload["bus_robust"].get("edge_pairs", [])]
    return components, closure_pairs, direct_pairs


def matched_candidates(nodes, components, target_pairs, heads):
    membership = {}
    for component_index, component in enumerate(components):
        for node in component:
            membership[node] = component_index
    pools = []
    for left, right, _ in target_pairs:
        layers = tuple(sorted((left // heads, right // heads)))
        candidates = []
        for a in nodes:
            for b in nodes:
                if a >= b or tuple(sorted((a // heads, b // heads))) != layers:
                    continue
                if membership.get(a, -1) >= 0 and membership.get(a) == membership.get(b):
                    continue
                candidates.append((a, b))
        pools.append(candidates)
    return pools


def enrichment(similarity, target_pairs, pools, replicates, rng):
    observed_values = np.asarray([similarity[a, b] for a, b, _ in target_pairs])
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        null[replicate] = np.mean([
            similarity[pair[0], pair[1]]
            for pool in pools
            for pair in [pool[int(rng.integers(len(pool)))]]
        ])
    observed = float(observed_values.mean())
    return {
        "pairs": len(observed_values),
        "component_mean": observed,
        "matched_mean": float(null.mean()),
        "enrichment": float(observed - null.mean()),
        "one_sided_p": float((1 + np.sum(null >= observed)) / (replicates + 1)),
        "component_median": float(np.median(observed_values)),
        "matched_q95": float(np.quantile(null, .95)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    dictionary = json.loads(Path(config["direction_dictionary"]).read_text())
    census = json.loads(Path(config["census_config"]).read_text())
    motif = json.loads(Path(config["motif_redteam"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index]))
                      for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32)
    negative = len(flags) - positive
    all_contrasts = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    competent = dictionary["competent_features"]
    feature_indices = [FEATURES.index(name) for name in competent]
    semantic = all_contrasts[:, feature_indices].astype(np.float32)
    # Reproduce the already-declared Gram-matched null exactly.
    permutation = np.random.default_rng(dictionary["config"]["seed"] + 2).permutation(len(flags))
    gram_matched = semantic[permutation]
    gram_error = float(np.linalg.norm(semantic.T @ semantic - gram_matched.T @ gram_matched)
                       / np.linalg.norm(semantic.T @ semantic))

    corpus = np.load(census["corpus"], mmap_mode="r")
    heldout_start = census["start_row"] + census["discovery_rows"]
    heldout = corpus[heldout_start:heldout_start + census["heldout_rows"]]
    # The direction-dictionary pilot used only four contexts per feature.  That
    # is enough to test recurrence of full residual directions, but not enough
    # to estimate a 162-head functional fingerprint.  Uniformly select fifty
    # positive held-out sites per competence-gated feature before computing any
    # head gradients.
    position_list = list(census["positions"])
    candidates = []
    for row_index in range(len(heldout)):
        next_ids = np.asarray(heldout[row_index, np.asarray(position_list) + 1], dtype=np.int64)
        for position_index, position in enumerate(position_list):
            candidates.append((row_index, int(position), int(next_ids[position_index])))
    rng_sites = np.random.default_rng(config["seed"])
    records = []
    for label, (feature, feature_index) in enumerate(zip(competent, feature_indices)):
        eligible = np.asarray([index for index, (_, _, token) in enumerate(candidates)
                               if flags[token, feature_index]], dtype=np.int64)
        chosen = rng_sites.choice(eligible, size=config["sites_per_feature"], replace=False)
        for local_index, candidate_index in enumerate(chosen):
            row_index, position, next_token = candidates[int(candidate_index)]
            current_token = int(heldout[row_index, position])
            records.append({
                "feature": feature, "feature_index": feature_index, "label": label,
                "selection": "uniform_a" if local_index < config["sites_per_feature"] // 2 else "uniform_b",
                "row": row_index, "position": position,
                "current_token": tokenizer.decode([current_token]),
                "next_token": tokenizer.decode([next_token]),
            })
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    layers, heads = model.config.n_layer, model.config.n_head
    gradients = {"semantic": [], "gram_matched": []}
    all_one_errors = []
    site_meta = []
    for start in range(0, len(records), config["batch_size"]):
        batch_records = records[start:start + config["batch_size"]]
        raw = np.stack([np.asarray(heldout[row["row"], :census["sequence_length"]], dtype=np.int64)
                        for row in batch_records])
        ids = torch.as_tensor(raw, device="cuda")
        positions = torch.as_tensor([row["position"] for row in batch_records], device="cuda")
        gates = torch.ones(len(batch_records), layers, heads, device="cuda", requires_grad=True)
        logits = gated_forward(model, ids, positions, gates)
        semantic_contrast = torch.as_tensor(
            np.stack([semantic[:, row["label"]] for row in batch_records]), device="cuda")
        null_contrast = torch.as_tensor(
            np.stack([gram_matched[:, row["label"]] for row in batch_records]), device="cuda")
        semantic_scores = (logits[:, :len(flags)] * semantic_contrast).sum(1)
        null_scores = (logits[:, :len(flags)] * null_contrast).sum(1)
        semantic_gradient = torch.autograd.grad(semantic_scores.sum(), gates, retain_graph=True)[0]
        null_gradient = torch.autograd.grad(null_scores.sum(), gates)[0]
        gradients["semantic"].append(semantic_gradient.detach().float().cpu().numpy())
        gradients["gram_matched"].append(null_gradient.detach().float().cpu().numpy())
        if start == 0:
            # Keep batch and sequence shapes identical: bfloat16 GEMM results
            # can legitimately differ when either shape changes.
            reference = native_forward_positions(model, ids, positions)
            all_one_errors.extend((reference - logits).abs().amax(1).float().cpu().tolist())
        site_meta.extend({key: row[key] for key in
                          ("feature", "label", "selection", "row", "position", "current_token", "next_token")}
                         for row in batch_records)
        print(f"anchor head gradients {min(start + len(batch_records), len(records))}/{len(records)}", flush=True)
    gradients = {name: np.concatenate(values) for name, values in gradients.items()}

    # Fifty independently sampled positive contexts per feature, split 25/25.
    split = np.asarray([0 if row["selection"] == "uniform_a" else 1 for row in records],
                       dtype=np.int64)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    arrays = {"labels": labels, "split": split}
    profiles = {}
    split_geometry = {}
    for objective, values in gradients.items():
        flat = values.reshape(len(values), -1)
        normalized_sites = unit_rows(flat)
        objective_profiles = []
        for side in (0, 1):
            profile = np.zeros((layers * heads, len(competent)), dtype=np.float64)
            for label in range(len(competent)):
                selected = (labels == label) & (split == side)
                profile[:, label] = normalized_sites[selected].mean(0)
            objective_profiles.append(profile)
            arrays[f"{objective}_profile_split{side}"] = profile.astype(np.float32)
        profiles[objective] = objective_profiles
        geometry_a = cosine_rows(objective_profiles[0])
        geometry_b = cosine_rows(objective_profiles[1])
        split_geometry[objective] = {
            "pairwise_similarity_pearson": pearson_triangle(geometry_a, geometry_b),
            "median_head_profile_cosine": float(np.median(np.sum(
                unit_rows(objective_profiles[0]) * unit_rows(objective_profiles[1]), axis=1))),
        }
    arrays.update({f"{name}_site_gradients": value.astype(np.float32)
                   for name, value in gradients.items()})

    components, target_pairs, direct_pairs = components_and_pairs(motif, heads)
    nodes = list(range(layers * heads))
    pools = matched_candidates(nodes, components, target_pairs, heads)
    direct_pools = matched_candidates(nodes, components, direct_pairs, heads)
    rng = np.random.default_rng(config["seed"])
    comparisons = {}
    direct_edge_redteam = {}
    combined_profiles = {}
    for objective, sides in profiles.items():
        combined = unit_rows((sides[0] + sides[1]) / 2)
        combined_profiles[objective] = combined
        comparisons[objective] = enrichment(
            combined @ combined.T, target_pairs, pools,
            config["matched_null_replicates"], rng,
        )
        direct_edge_redteam[objective] = enrichment(
            combined @ combined.T, direct_pairs, direct_pools,
            config["matched_null_replicates"], rng,
        )

    # Interpret components only after the frozen comparison.  Absolute signed
    # sensitivity is used so both promotion and suppression count as support.
    component_summaries = []
    semantic_profile = combined_profiles["semantic"]
    for component_index, component in enumerate(components):
        score = np.mean(np.abs(semantic_profile[component]), axis=0)
        order = np.argsort(score)[::-1][:5]
        component_summaries.append({
            "component": component_index,
            "members": [{"layer": node // heads, "head": node % heads} for node in component],
            "top_supervised_anchors": [
                {"feature": competent[index], "score": float(score[index])} for index in order
            ],
            "effective_feature_support": float(score.sum() ** 2 / max(np.square(score).sum(), 1e-30)),
        })

    # Prospective finite validation: choose large semantic gradients without
    # looking at finite effects, then downscale exactly one head by 10%.
    semantic_flat = gradients["semantic"].reshape(len(records), -1)
    chosen = np.argsort(np.max(np.abs(semantic_flat), axis=1))[::-1][:config["finite_validation_sites"]]
    finite_rows = []
    for site_index in chosen:
        head_node = int(np.argmax(np.abs(semantic_flat[site_index])))
        row = records[int(site_index)]
        raw = np.asarray(heldout[row["row"], :census["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")[None]
        positions = torch.as_tensor([row["position"]], device="cuda")
        gates = torch.ones(1, layers, heads, device="cuda")
        contrast = torch.as_tensor(semantic[:, row["label"]], device="cuda")
        with torch.no_grad():
            base = (gated_forward(model, ids, positions, gates)[0, :len(flags)] * contrast).sum()
            gates[0, head_node // heads, head_node % heads] = config["finite_gate_keep"]
            edited = (gated_forward(model, ids, positions, gates)[0, :len(flags)] * contrast).sum()
        predicted = (config["finite_gate_keep"] - 1) * semantic_flat[site_index, head_node]
        finite_rows.append({
            "feature": row["feature"], "site_index": int(site_index),
            "layer": head_node // heads, "head": head_node % heads,
            "gradient": float(semantic_flat[site_index, head_node]),
            "predicted_change": float(predicted), "finite_change": float(edited - base),
        })
    predicted = np.asarray([row["predicted_change"] for row in finite_rows])
    finite = np.asarray([row["finite_change"] for row in finite_rows])
    finite_validation = {
        "sites": len(finite_rows),
        "pearson": float(np.corrcoef(predicted, finite)[0, 1]),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(finite))),
        "median_absolute_relative_error": float(np.median(np.abs(predicted - finite)
            / np.maximum(np.abs(finite), 1e-8))),
        "records": finite_rows,
    }

    semantic_enrichment = comparisons["semantic"]["enrichment"]
    null_enrichment = comparisons["gram_matched"]["enrichment"]
    maximum_forward_error = max(all_one_errors)
    hypotheses = {
        "A0_exact_forward": {"supported": maximum_forward_error <= .02,
                             "maximum_absolute_logit_error": maximum_forward_error},
        "A1_local_causal_measure": {"supported": finite_validation["pearson"] >= .95 and finite_validation["sign_agreement"] >= .9,
                                    "pearson": finite_validation["pearson"], "sign_agreement": finite_validation["sign_agreement"]},
        "A2_supervised_profiles_replicate": {"supported": split_geometry["semantic"]["pairwise_similarity_pearson"] >= .5,
                                              **split_geometry["semantic"]},
        "A3_weight_motifs_recover_supervised_function": {"supported": comparisons["semantic"]["enrichment"] > 0 and comparisons["semantic"]["one_sided_p"] < .01,
                                                          **comparisons["semantic"]},
        "A4_recovery_is_semantic": {"supported": semantic_enrichment - null_enrichment >= .03,
                                     "semantic_minus_null_enrichment": semantic_enrichment - null_enrichment},
    }
    np.savez_compressed(ARRAYS, **arrays)
    payload = {
        "config": config, "method": {
            "labels": "32 competence-gated members of the predeclared 50-label census, with 50 uniformly sampled positive contexts per label",
            "causal_fingerprint": "d(feature logit contrast)/d(multiplicative attention-head write gate)",
            "discovery_leakage": "none: bus-robust weight components were frozen before this comparison",
            "matched_control": "non-component head pairs with the identical unordered layer pair",
            "semantic_null": "one shared vocabulary permutation preserving the complete objective Gram matrix",
        },
        "features": competent, "sites_per_feature": config["sites_per_feature"],
        "site_count": len(records), "gram_relative_error": gram_error,
        "all_one_maximum_absolute_logit_error": maximum_forward_error,
        "split_geometry": split_geometry, "comparisons": comparisons,
        "direct_edge_redteam": direct_edge_redteam,
        "finite_validation": finite_validation, "hypotheses": hypotheses,
        "component_summaries": component_summaries, "arrays": str(ARRAYS),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Supervised anchors versus frozen weight motifs", "",
        "The earlier 50-label run produced 32 competence-gated semantic targets, not 50 established circuits. Here those targets are used as external anchors for the already-frozen, label-blind attention-head components. Each head is fingerprinted by the signed derivative of the target logit contrast with respect to a multiplicative write gate on held-out positive sites.", "",
        "## Locked results", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Anchor comparison](bilin18_supervised_anchor_motif_comparison.png)", "",
              "| Objective | Split geometry r | Median per-head split cosine | Component cosine | Layer-matched cosine | Enrichment | p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for objective in ("semantic", "gram_matched"):
        geometry = split_geometry[objective]; comparison = comparisons[objective]
        lines.append(f"| {objective} | {geometry['pairwise_similarity_pearson']:.3f} | {geometry['median_head_profile_cosine']:.3f} | {comparison['component_mean']:.3f} | {comparison['matched_mean']:.3f} | {comparison['enrichment']:+.3f} | {comparison['one_sided_p']:.4f} |")
    lines += ["", "The locked cluster test uses all within-component pairs, matching the wording fixed before evaluation. A stricter post-hoc direct-edge red team gives: "
              + "; ".join(f"{name} enrichment={direct_edge_redteam[name]['enrichment']:+.3f}, p={direct_edge_redteam[name]['one_sided_p']:.4f}" for name in ("semantic", "gram_matched")) + ".", "",
              "Finite 10% gate validation: "
              f"r={finite_validation['pearson']:.3f}, sign agreement={finite_validation['sign_agreement']:.3f}, "
              f"median absolute relative error={finite_validation['median_absolute_relative_error']:.3f}. "
              f"All-one maximum logit error={maximum_forward_error:.3e}.", "",
              "## Component interpretations from supervised anchors", "",
              "These names are post-hoc handles, not cluster definitions. Broad or semantically mixed anchor lists can be consistent with a reusable routing verb applied to different objects.", ""]
    for row in component_summaries:
        members = ", ".join(f"L{member['layer']}H{member['head']}" for member in row["members"])
        anchors = ", ".join(f"{anchor['feature']} ({anchor['score']:.3f})" for anchor in row["top_supervised_anchors"])
        lines.append(f"- Component {row['component']} [{members}], effective support {row['effective_feature_support']:.1f}: {anchors}.")
    lines += ["", "## What this does and does not validate", "",
              "- Positive recovery means independently weight-similar heads also have similar local causal profiles over known model capabilities.",
              "- Failure of semantic purity does not by itself falsify a shared computational verb: the same routing operation can act on unrelated semantic payloads.",
              "- The gate derivative is local. The finite 10% test checks that it predicts nearby interventions; it is not a claim about deleting an entire head.",
              "- The 32 labels are a deliberately limited human ontology and overlap lexically. They are silver-standard anchors, not the model's ground-truth latent variables."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = ["semantic", "gram_matched"]
    x = np.arange(2)
    axes[0].bar(x - .18, [comparisons[name]["component_mean"] for name in names], .36, label="component pairs")
    axes[0].bar(x + .18, [comparisons[name]["matched_mean"] for name in names], .36, label="layer-matched")
    axes[0].set_xticks(x, names); axes[0].set_ylabel("Mean causal-profile cosine")
    axes[0].set_title("Do frozen weight motifs recover supervised function?"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].scatter(predicted, finite)
    lo = min(predicted.min(), finite.min()); hi = max(predicted.max(), finite.max())
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    axes[1].set(xlabel="Gate-gradient prediction", ylabel="Finite 10% change", title="Local causal validation")
    axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "split_geometry": split_geometry,
                      "comparisons": comparisons, "direct_edge_redteam": direct_edge_redteam,
                      "finite_validation": finite_validation}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
