#!/usr/bin/env python3
"""Red-team finite-sketch stability of the early operator-tensor vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import (
    centered_r2,
    collect_states,
    feature_metrics,
    operator_modes,
    operator_probe_vectors,
    pca_reconstruct,
    reconstruct,
    subspace_overlap,
)
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_operator_probe_convergence.json")
OUTPUT = TENSOR_ROOT / "bilin18_operator_probe_convergence.json"
REPORT = TENSOR_ROOT / "BILIN18_OPERATOR_PROBE_CONVERGENCE.md"
FIGURE = TENSOR_ROOT / "bilin18_operator_probe_convergence.png"
BASES = TENSOR_ROOT / "bilin18_operator_probe_convergence_bases.pt"


def linear_sum_assignment_maximum(score):
    """Exact square maximum-weight assignment via the Hungarian algorithm."""
    score = np.asarray(score, dtype=np.float64)
    size = score.shape[0]
    if score.shape != (size, size):
        raise ValueError("assignment matrix must be square")
    cost = score.max() - score
    u = np.zeros(size + 1); v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=np.int64); way = np.zeros(size + 1, dtype=np.int64)
    for i in range(1, size + 1):
        p[0] = i
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, size + 1):
                if used[j]: continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = current, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0: break
    columns = np.arange(size, dtype=np.int64)
    rows = p[1:] - 1
    order = np.argsort(rows)
    return rows[order], columns[order]


def covariance_geometry(states):
    mean = states.mean(0)
    centered = states - mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    order = values.argsort(descending=True)
    return mean, vectors[:, order], values[order].clamp_min(1e-8).sqrt()


def semantic_axis_profiles(train, labels, eligible):
    train = (train - train.mean(0)) / train.std(0).clamp_min(1e-4)
    profiles = []
    for feature in eligible:
        selected = labels[:, feature].bool()
        profiles.append(train[selected].mean(0) - train[~selected].mean(0))
    return torch.stack(profiles).T


def rotation_p95(train, test, train_labels, test_labels, current, eligible, rank, rounds, generator):
    values = []
    for _ in range(rounds):
        rotation = torch.linalg.qr(torch.randn(rank, rank, generator=generator, device="cuda")).Q
        row = feature_metrics(
            train @ rotation, test @ rotation, train_labels, test_labels,
            current, eligible, rank, include_within=False,
        )
        values.append(row["single_median"])
    return float(np.quantile(values, .95))


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    source = {**source, "layers": config["layers"]}
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    competence = json.loads(Path(source["competence_results"]).read_text())
    eligible = [FEATURES.index(row["feature"]) for row in competence["features"] if row["competent"]]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("collect convergence panels", flush=True)
    discovery, discovery_labels, _ = collect_states(model, source["corpus"], source["discovery_start"], source["rows_per_split"], source, flags)
    evaluation, evaluation_labels, current = collect_states(model, source["corpus"], source["evaluation_start"], source["rows_per_split"], source, flags)
    train_labels = torch.from_numpy(discovery_labels).cuda(); test_labels = torch.from_numpy(evaluation_labels).cuda()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 999)
    results, basis_store = {}, {}
    for layer in config["layers"]:
        print("probe convergence layer", layer, flush=True)
        train = torch.from_numpy(discovery[layer]).cuda(); test = torch.from_numpy(evaluation[layer]).cuda()
        mean, vectors, rms = covariance_geometry(train)
        evaluation_probe, _, _ = operator_probe_vectors(model.blocks[layer], config["evaluation_probes"], config["seed"] + 1000 + layer)
        exact = test @ evaluation_probe.T
        pca = pca_reconstruct(test, mean, vectors, config["rank"]) @ evaluation_probe.T
        pca_r2 = centered_r2(exact, pca)
        curves, final = {}, None
        for count in config["probe_counts"]:
            first, _, _ = operator_probe_vectors(model.blocks[layer], count, config["seed"] + 20 * layer)
            second, _, _ = operator_probe_vectors(model.blocks[layer], count, config["seed"] + 20 * layer + 1)
            singular_a, modes_a = operator_modes(vectors, rms, first)
            singular_b, modes_b = operator_modes(vectors, rms, second)
            prediction = reconstruct(test, mean, vectors, rms, modes_a, config["rank"]) @ evaluation_probe.T
            curves[str(count)] = {
                "overlap": subspace_overlap(modes_a[:, :config["rank"]], modes_b[:, :config["rank"]]),
                "heldout_r2": centered_r2(exact, prediction),
                "top64_energy": float(singular_a[:config["rank"]].square().sum() / singular_a.square().sum()),
            }
            if count == config["probe_counts"][-1]: final = (modes_a, modes_b)
        modes_a, modes_b = final
        whitened_train = ((train - mean) @ vectors) / rms[None]
        whitened_test = ((test - mean) @ vectors) / rms[None]
        semantic = {}
        coordinates = {}
        for name, modes in (("first", modes_a), ("second", modes_b)):
            train_coordinates = whitened_train @ modes[:, :config["rank"]]
            test_coordinates = whitened_test @ modes[:, :config["rank"]]
            coordinates[name] = train_coordinates
            row = feature_metrics(train_coordinates, test_coordinates, train_labels, test_labels, current, eligible, config["rank"])
            row["rotation_p95"] = rotation_p95(train_coordinates, test_coordinates, train_labels, test_labels, current, eligible, config["rank"], source["random_rotations"], generator)
            semantic[name] = row
        overlap = (modes_a[:, :config["rank"]].T @ modes_b[:, :config["rank"]]).abs().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment_maximum(overlap)
        profiles_a = semantic_axis_profiles(coordinates["first"], train_labels, eligible).cpu().numpy()
        profiles_b = semantic_axis_profiles(coordinates["second"], train_labels, eligible).cpu().numpy()
        correlations = []
        for a, b in zip(row_ind, col_ind):
            first_profile, second_profile = profiles_a[a], profiles_b[b]
            if float(overlap[a, b]) < 1e-8: continue
            if np.dot(first_profile, second_profile) < 0: second_profile = -second_profile
            correlations.append(float(np.corrcoef(first_profile, second_profile)[0, 1]))
        results[str(layer)] = {
            "state_pca_rank64_r2": pca_r2,
            "curve": curves,
            "semantics": semantic,
            "matched_axis_cosine_median": float(np.median(overlap[row_ind, col_ind])),
            "semantic_profile_correlation_mean": float(np.nanmean(correlations)),
        }
        basis_store[str(layer)] = {
            "mean": mean.cpu(), "state_vectors": vectors.cpu(), "state_rms": rms.cpu(),
            "operator_modes_first": modes_a.cpu(), "operator_modes_second": modes_b.cpu(),
            "operator_singular_first": singular_a.cpu(), "operator_singular_second": singular_b.cpu(),
        }
    maximum = str(config["probe_counts"][-1])
    hypotheses = {
        "C0_probe_overlap_converges": {"supported": all(row["curve"][maximum]["overlap"] >= .80 for row in results.values())},
        "C1_operator_reconstruction_beats_pca_after_convergence": {"supported": all(row["curve"][maximum]["heldout_r2"] > row["state_pca_rank64_r2"] for row in results.values())},
        "C2_semantic_axis_advantage_replicates": {"supported": all(all(cell["single_median"] - cell["rotation_p95"] >= .02 for cell in row["semantics"].values()) for row in results.values())},
        "C3_semantic_axis_identity_replicates": {"supported": all(row["semantic_profile_correlation_mean"] >= .70 for row in results.values())},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    torch.save(basis_store, BASES)
    lines = ["# Operator-tensor probe convergence red team", "", "The early operator vocabulary is refit with 256--4096 independent Kronecker measurements per layer. Every basis is evaluated against a third, 4096-probe sketch on disjoint FineWeb states. At the maximum budget, semantic-axis scores must replicate in two independently fitted bases and matched axes must retain their semantic profiles.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Probe convergence](bilin18_operator_probe_convergence.png)", "", "| Layer | Overlap 256/512/1024/2048/4096 | Heldout R2 256/512/1024/2048/4096 | PCA64 | Semantic one A/B vs rotations | Matched cosine / semantic profile r |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; curve = row["curve"]
        overlaps = "/".join(f"{curve[str(count)]['overlap']:.3f}" for count in config["probe_counts"])
        r2s = "/".join(f"{curve[str(count)]['heldout_r2']:.3f}" for count in config["probe_counts"])
        a, b = row["semantics"]["first"], row["semantics"]["second"]
        lines.append(f"| {layer} | {overlaps} | {r2s} | {row['state_pca_rank64_r2']:.3f} | {a['single_median']:.3f}/{b['single_median']:.3f} vs {a['rotation_p95']:.3f}/{b['rotation_p95']:.3f} | {row['matched_axis_cosine_median']:.3f}/{row['semantic_profile_correlation_mean']:.3f} |")
    lines += ["", "## Guardrail", "", "A converged Frobenius operator basis would be canonical only for the local MLP Jacobian under FineWeb covariance. Semantic-profile replication is stronger than a one-run axis AUROC, but it remains supervised evidence about a predeclared ontology. Failure to beat state PCA means tensor weighting did not provide a shorter predictive code, even if individual axes are reproducible."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in config["layers"]:
        row = results[str(layer)]; counts = config["probe_counts"]
        axes[0].plot(counts, [row["curve"][str(count)]["overlap"] for count in counts], marker="o", label=f"L{layer}")
        axes[1].plot(counts, [row["curve"][str(count)]["heldout_r2"] for count in counts], marker="o", label=f"L{layer} operator")
        axes[1].axhline(row["state_pca_rank64_r2"], linestyle="--", alpha=.5)
    axes[0].set(xscale="log", xlabel="Kronecker probes", ylabel="Independent top-64 overlap", title="Does the operator basis converge?"); axes[0].grid(alpha=.2); axes[0].legend()
    axes[1].set(xscale="log", xlabel="Kronecker probes", ylabel="Held-out centered R2", title="Operator basis versus PCA dashed lines"); axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
