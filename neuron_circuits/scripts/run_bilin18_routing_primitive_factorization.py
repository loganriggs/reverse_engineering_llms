#!/usr/bin/env python3
"""Factor routing profiles into reusable variables and read-function uses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_model_derived_source_ontology import weighted_nmi
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports


CONFIG = Path("configs/bilin18_routing_primitive_factorization.json")
PROFILE_SOURCE = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
ASSIGNMENT_SOURCE = TENSOR_ROOT / "bilin18_causal_source_ontology_assignments.npz"
OUTPUT = TENSOR_ROOT / "bilin18_routing_primitive_factorization.json"
FACTORS = TENSOR_ROOT / "bilin18_routing_primitive_factors.pt"
REPORT = TENSOR_ROOT / "BILIN18_ROUTING_PRIMITIVE_FACTORIZATION.md"


def weighted_nmf(matrix, weights, rank, iterations, restarts, seed):
    observed = (weights > 0) & (matrix.square().sum(1) > 0)
    x = matrix[observed].float()
    row_scale = weights[observed].float().sqrt().clamp_min(1e-12)
    target = x * row_scale[:, None]
    best = None
    for restart in range(restarts):
        generator = torch.Generator(device=matrix.device).manual_seed(seed + restart)
        w = torch.rand(len(target), rank, generator=generator, device=matrix.device).clamp_min(1e-4)
        h = torch.rand(rank, target.shape[1], generator=generator, device=matrix.device).clamp_min(1e-4)
        for _ in range(iterations):
            h *= (w.T @ target) / (w.T @ w @ h).clamp_min(1e-8)
            w *= (target @ h.T) / (w @ (h @ h.T)).clamp_min(1e-8)
            scale = h.norm(dim=1).clamp_min(1e-8)
            h /= scale[:, None]; w *= scale[None]
        error = float((target - w @ h).square().sum() / target.square().sum().clamp_min(1e-30))
        if best is None or error < best[2]:
            membership = torch.zeros(len(matrix), rank, device=matrix.device)
            membership[observed] = w / row_scale[:, None]
            best = (membership, h, error)
    return best


def greedy_alignment(first, second):
    similarity = F.normalize(first, dim=1) @ F.normalize(second, dim=1).T
    available_first = set(range(len(first))); available_second = set(range(len(second)))
    pairs = []
    while available_first:
        best = max(((float(similarity[i, j]), i, j) for i in available_first for j in available_second), key=lambda row: row[0])
        pairs.append(best); available_first.remove(best[1]); available_second.remove(best[2])
    pairs.sort(key=lambda row: row[1])
    return pairs


def weighted_spectrum(matrix, weights):
    valid = weights > 0
    x = matrix[valid].float(); w = weights[valid].float(); w /= w.sum()
    mean = (x * w[:, None]).sum(0)
    weighted = (x - mean) * w.sqrt()[:, None]
    values = torch.linalg.svdvals(weighted).square()
    fraction = values.cumsum(0) / values.sum().clamp_min(1e-30)
    ranks = {str(level): int(torch.searchsorted(fraction, level).item() + 1) for level in (.5, .8, .9, .95)}
    participation = float(values.sum().square() / values.square().sum().clamp_min(1e-30))
    return ranks, participation


def support_at_fraction(values, fraction=.9):
    ordered = values.sort(dim=1, descending=True).values
    cumulative = ordered.cumsum(1) / ordered.sum(1, keepdim=True).clamp_min(1e-30)
    return (cumulative < fraction).sum(1) + 1


def decode_coordinate(index, config):
    distances = len(config["distance_bins"]) - 1; per_layer = 9 * distances
    layer = index // per_layer; remainder = index % per_layer
    head = remainder // distances; distance = remainder % distances
    return f"L{layer} H{head} d[{config['distance_bins'][distance]},{config['distance_bins'][distance + 1]})"


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    profiles = torch.load(PROFILE_SOURCE, map_location="cpu", weights_only=True)
    prior_assignments = np.load(ASSIGNMENT_SOURCE)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    counts_a = profiles["a"]["counts"].float(); counts_b = profiles["b"]["counts"].float()
    weights = (counts_a + counts_b).sqrt()
    matrix_a = profiles["a"]["routing"].float().to("cuda"); matrix_b = profiles["b"]["routing"].float().to("cuda")
    weights_cuda = weights.to("cuda")
    print("fit NMF A", flush=True)
    membership_a, functions_a, error_a = weighted_nmf(matrix_a, weights_cuda, config["rank"], config["nmf_iterations"], config["nmf_restarts"], config["seed"])
    print("fit NMF B", flush=True)
    membership_b, functions_b, error_b = weighted_nmf(matrix_b, weights_cuda, config["rank"], config["nmf_iterations"], config["nmf_restarts"], config["seed"] + 100)
    pairs = greedy_alignment(functions_a, functions_b)
    permutation = torch.as_tensor([row[2] for row in pairs], device="cuda")
    aligned_b = membership_b[:, permutation]
    function_cosines = [row[0] for row in pairs]
    normalized_a = F.normalize(membership_a, dim=1); normalized_b = F.normalize(aligned_b, dim=1)
    common = (counts_a > 0) & (counts_b > 0)
    common_weight = (counts_a * counts_b).sqrt().to("cuda") * common.to("cuda")
    membership_cosine = float(((normalized_a * normalized_b).sum(1) * common_weight).sum() / common_weight.sum().clamp_min(1e-30))
    assignment_a = membership_a.argmax(1).cpu().numpy().astype(np.int64)
    assignment_b_unaligned = membership_b.argmax(1)
    inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(len(permutation), device="cuda")
    assignment_b = inverse[assignment_b_unaligned].cpu().numpy().astype(np.int64)
    hard_nmi = weighted_nmi(assignment_a, assignment_b, weights.numpy(), config["rank"])
    spectrum_a, participation_a = weighted_spectrum(matrix_a, weights_cuda)
    spectrum_b, participation_b = weighted_spectrum(matrix_b, weights_cuda)
    token_support = support_at_fraction(membership_a).float()
    function_support = support_at_fraction(functions_a).float()
    token_support_median = float(token_support[(counts_a > 0).to("cuda")].median())
    function_support_median = float(function_support.median())
    torch.save({"membership_a": membership_a.cpu(), "membership_b": membership_b.cpu(), "functions_a": functions_a.cpu(), "functions_b": functions_b.cpu(), "permutation_b_to_a": permutation.cpu()}, FACTORS)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for name, assignment in (("nmf_a", assignment_a), ("nmf_b", assignment_b)):
        print("evaluate NMF ontology", name, flush=True)
        supports = select_supports(model, discovery, assignment, config)
        row, error = evaluate(model, evaluation, assignment, config, supports)
        results[name] = row["5"]; maximum_error = max(maximum_error, error)
    score = {name: results[name]["methods"]["fine128"]["kl_recovery"] for name in results}
    top1 = {name: results[name]["methods"]["fine128"]["top1_agreement"] for name in results}
    hypotheses = {
        "F1_routing_profiles_are_low_dimensional": {"supported": spectrum_a["0.9"] <= 64 and spectrum_b["0.9"] <= 64, "rank90_a": spectrum_a["0.9"], "rank90_b": spectrum_b["0.9"]},
        "F2_read_function_primitives_are_stable": {"supported": np.mean(function_cosines) >= .80, "mean_cosine": float(np.mean(function_cosines)), "median_cosine": float(np.median(function_cosines))},
        "F3_soft_variable_memberships_are_stable": {"supported": membership_cosine >= .70, "weighted_cosine": membership_cosine},
        "F4_hard_variable_assignments_are_more_stable": {"supported": hard_nmi >= .55, "weighted_nmi": hard_nmi},
        "F5_hard_nmf_variables_are_causally_useful": {"supported": max(score.values()) >= .90},
        "F6_hard_nmf_variables_are_prediction_useful": {"supported": max(top1.values()) >= .65},
        "F7_primitive_uses_are_sparse": {"supported": token_support_median <= 4 and function_support_median <= 32, "token_components90_median": token_support_median, "function_coordinates90_median": function_support_median},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    primitive_rows = []
    frequency = weights.numpy()
    for component in range(config["rank"]):
        token_score = membership_a[:, component].cpu().numpy() * frequency
        tokens = np.argsort(token_score)[::-1][:16]
        coordinates = functions_a[component].argsort(descending=True)[:10].cpu().numpy()
        primitive_rows.append({
            "component": component,
            "matched_function_cosine": function_cosines[component],
            "top_tokens": [tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t") for index in tokens],
            "top_read_functions": [decode_coordinate(int(index), config) for index in coordinates],
        })
    payload = {"config": config, "hypotheses": hypotheses, "nmf_error": {"a": error_a, "b": error_b}, "spectrum": {"a": spectrum_a, "b": spectrum_b, "participation_a": participation_a, "participation_b": participation_b}, "function_cosines": function_cosines, "membership_cosine": membership_cosine, "hard_nmi": hard_nmi, "primitive_rows": primitive_rows, "results": results, "prior_kmeans_nmi": weighted_nmi(prior_assignments["operator_profile_a"], prior_assignments["operator_profile_b"], weights.numpy(), config["rank"])}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Routing variables and read-function primitives", "",
        "The nonnegative routing profile is factorized as `token memberships × read-function uses`. A variable can therefore be reused by several layer/head/distance functions, and a token can participate in several variables. Hard argmax assignments are evaluated only as a lossy bridge to the existing typed-edge executor.", "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", f"Weighted rank-90: `{spectrum_a['0.9']}` / `{spectrum_b['0.9']}`; participation ratio: `{participation_a:.1f}` / `{participation_b:.1f}`.", f"Mean/median matched function cosine: `{np.mean(function_cosines):.3f}` / `{np.median(function_cosines):.3f}`. Soft membership cosine: `{membership_cosine:.3f}`. Hard assignment NMI: `{hard_nmi:.3f}`.", "", "| Hard view | KL recovery | CE delta | Top-1 |", "|---|---:|---:|---:|"]
    for name in ("nmf_a", "nmf_b"):
        value = results[name]["methods"]["fine128"]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    lines += ["", "## Candidate variable/function primitives", ""]
    for row in primitive_rows:
        tokens = ", ".join(repr(token) for token in row["top_tokens"][:8]); functions = ", ".join(row["top_read_functions"][:5])
        lines.append(f"- **V{row['component']}** (split function cosine {row['matched_function_cosine']:.2f}): tokens {tokens}; reads {functions}.")
    lines += ["", "## MDL interpretation", "", "This decomposition separates four potential code-length charges: declare a variable (one token-membership column), declare a read function (one layer/head/distance pattern), specify the variable–function links, and invoke the resulting route in an example. The current report measures factor rank and 90%-mass support counts but does not yet claim a calibrated bit-level MDL. A primitive is readable only if its token membership and function uses remain stable and admit a short description; causal fidelity alone is insufficient.", "", "## Guardrails", "", "NMF is non-convex and the greedy component matching is not an optimal assignment. Magnitude profiles omit signs and contextual token-state changes. The hard causal evaluation discards all but the largest membership per token, so it can underestimate a valid multiplexed variable representation. Conversely, a good hard score would not prove that NMF components are semantic. The next test should execute soft memberships directly and compare description length at matched behavior."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "score": score, "top1": top1}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
