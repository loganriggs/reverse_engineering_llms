#!/usr/bin/env python3
"""Find a weight-derived vocabulary for early state-conditioned MLP operators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features, within_token_auc
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_early_operator_tensor_vocabulary.json")
OUTPUT = TENSOR_ROOT / "bilin18_early_operator_tensor_vocabulary.json"
REPORT = TENSOR_ROOT / "BILIN18_EARLY_OPERATOR_TENSOR_VOCABULARY.md"
FIGURE = TENSOR_ROOT / "bilin18_early_operator_tensor_vocabulary.png"
BASES = TENSOR_ROOT / "bilin18_early_operator_tensor_vocabulary.pt"


@torch.inference_mode()
def collect_states(model, corpus, start, rows, config, flags):
    source = np.load(corpus, mmap_mode="r")
    panel = source[start:start + rows]
    positions = np.asarray(config["positions"])
    total = len(panel) * len(positions)
    dimension = model.config.n_embd
    states = {
        layer: np.empty((total, dimension), dtype=np.float32)
        for layer in config["layers"]
    }
    labels = np.empty((total, len(FEATURES)), dtype=np.uint8)
    current = np.empty(total, dtype=np.int32)
    offset = 0
    for row_start in range(0, len(panel), config["batch_size"]):
        raw = np.asarray(
            panel[row_start:row_start + config["batch_size"], :config["sequence_length"] + 1],
            dtype=np.int64,
        )
        ids = torch.as_tensor(raw[:, :config["sequence_length"]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x, value_bus = x0, None
        for layer, block in enumerate(model.blocks[:max(config["layers"]) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
            pre = mixed + attention
            normalized = F.rms_norm(pre, (dimension,))
            if layer in states:
                selected = normalized[:, positions].flatten(0, 1).float().cpu().numpy()
                states[layer][offset:offset + len(selected)] = selected
            product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
            x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        count = len(raw) * len(positions)
        current[offset:offset + count] = raw[:, positions].reshape(-1)
        labels[offset:offset + count] = flags[raw[:, positions + 1].reshape(-1)]
        offset += count
    return states, labels, current


@torch.inference_mode()
def operator_probe_vectors(block, count, seed):
    """Rows y satisfy z@y = u^T J_MLP(z)q for random Kronecker probes u,q."""
    dimension = block.mlp.Down.weight.shape[0]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    device = down.device
    generator = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(count, dimension, generator=generator, device=device) / np.sqrt(dimension)
    q = torch.randn(count, dimension, generator=generator, device=device) / np.sqrt(dimension)
    output = u @ down
    left_q = q @ left.T
    right_q = q @ right.T
    return (output * left_q) @ right + (output * right_q) @ left, u, q


def centered_r2(target, prediction):
    centered = target - target.mean(0, keepdim=True)
    return float(1 - (target - prediction).square().sum() / centered.square().sum().clamp_min(1e-30))


def cosine_mean(target, prediction):
    numerator = (target * prediction).sum(1)
    denominator = target.norm(dim=1) * prediction.norm(dim=1)
    return float((numerator / denominator.clamp_min(1e-30)).mean())


def subspace_overlap(first, second):
    rank = first.shape[1]
    return float((first.T @ second).square().sum() / rank)


def fit_geometry(states, probe_vectors):
    mean = states.mean(0)
    centered = states - mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    order = values.argsort(descending=True)
    values, vectors = values[order].clamp_min(1e-8), vectors[:, order]
    rms = values.sqrt()
    singular, modes = operator_modes(vectors, rms, probe_vectors)
    return mean, vectors, rms, singular, modes


def operator_modes(vectors, rms, probe_vectors):
    natural_map = probe_vectors @ (vectors * rms[None])
    _, singular, vh = torch.linalg.svd(natural_map, full_matrices=False)
    modes = vh.T
    return singular, modes


def reconstruct(states, mean, vectors, rms, modes, rank):
    whitened = ((states - mean) @ vectors) / rms[None]
    selected = modes[:, :rank]
    recovered = (whitened @ selected) @ selected.T
    return mean + (recovered * rms[None]) @ vectors.T


def pca_reconstruct(states, mean, vectors, rank):
    selected = vectors[:, :rank]
    return mean + ((states - mean) @ selected) @ selected.T


def feature_metrics(discovery, evaluation, discovery_labels, evaluation_labels, current_ids, eligible, rank, include_within=True):
    mean = discovery.mean(0)
    std = discovery.std(0).clamp_min(1e-4)
    train = (discovery - mean) / std
    test = (evaluation - mean) / std
    single, coalition, within = [], [], []
    for feature in eligible:
        labels = discovery_labels[:, feature].bool()
        gap = train[labels].mean(0) - train[~labels].mean(0)
        order = gap.abs().argsort(descending=True)
        one = order[:1]
        many = order[:rank]
        one_score = test[:, one] @ gap[one]
        many_score = test[:, many] @ gap[many]
        target = evaluation_labels[:, feature].cpu().numpy()
        single.append(float(auc(target, one_score.cpu().numpy())))
        coalition.append(float(auc(target, many_score.cpu().numpy())))
        if include_within:
            within.append(within_token_auc(target, many_score.cpu().numpy(), current_ids)[0])
    return {
        "single_median": float(np.median(single)),
        "coalition_median": float(np.median(coalition)),
        "within_token_median": float(np.median([value for value in within if value is not None])) if within else None,
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    competence = json.loads(Path(config["competence_results"]).read_text())
    eligible = [FEATURES.index(row["feature"]) for row in competence["features"] if row["competent"]]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    print("collect operator-vocabulary discovery", flush=True)
    discovery, discovery_labels, _ = collect_states(
        model, config["corpus"], config["discovery_start"], config["rows_per_split"], config, flags,
    )
    print("collect operator-vocabulary evaluation", flush=True)
    evaluation, evaluation_labels, evaluation_current = collect_states(
        model, config["corpus"], config["evaluation_start"], config["rows_per_split"], config, flags,
    )

    results, bases = {}, {}
    random_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 9000)
    maximum_identity_error = 0.0
    decoded_directions = {}
    for layer in config["layers"]:
        print("operator tensor layer", layer, flush=True)
        block = model.blocks[layer]
        train_x = torch.from_numpy(discovery[layer]).cuda()
        test_x = torch.from_numpy(evaluation[layer]).cuda()
        first, u, q = operator_probe_vectors(block, config["probe_count"], config["seed"] + 10 * layer)
        second, _, _ = operator_probe_vectors(block, config["probe_count"], config["seed"] + 10 * layer + 1)
        mean, vectors, rms, singular, modes = fit_geometry(train_x, first)
        _, _, _, singular_2, modes_2 = fit_geometry(train_x, second)

        # Check the defining tensor identity explicitly on four probes/states.
        for index in range(4):
            z = test_x[index]
            left = block.mlp.Left.weight.float() @ z
            right = block.mlp.Right.weight.float() @ z
            jacobian = block.mlp.Down.weight.float() @ (
                right[:, None] * block.mlp.Left.weight.float()
                + left[:, None] * block.mlp.Right.weight.float()
            )
            direct = u[index] @ jacobian @ q[index]
            implicit = z @ first[index]
            maximum_identity_error = max(
                maximum_identity_error,
                float((direct - implicit).abs() / direct.abs().clamp_min(1e-30)),
            )

        exact_sketch = test_x @ second.T
        rank_rows = {}
        for rank in config["ranks"]:
            operator_prediction = reconstruct(test_x, mean, vectors, rms, modes, rank) @ second.T
            pca_prediction = pca_reconstruct(test_x, mean, vectors, rank) @ second.T
            rank_rows[str(rank)] = {
                "operator_r2": centered_r2(exact_sketch, operator_prediction),
                "operator_cosine": cosine_mean(exact_sketch, operator_prediction),
                "state_pca_r2": centered_r2(exact_sketch, pca_prediction),
            }

        random_r2 = []
        whitened_test = ((test_x - mean) @ vectors) / rms[None]
        for _ in range(config["random_subspaces"]):
            random = torch.randn(
                model.config.n_embd, config["semantic_rank"], generator=random_generator, device="cuda",
            )
            random = torch.linalg.qr(random, mode="reduced").Q
            recovered = (whitened_test @ random) @ random.T
            prediction = (mean + (recovered * rms[None]) @ vectors.T) @ second.T
            random_r2.append(centered_r2(exact_sketch, prediction))

        rank = config["semantic_rank"]
        operator_train = (((train_x - mean) @ vectors) / rms[None]) @ modes[:, :rank]
        operator_test = (((test_x - mean) @ vectors) / rms[None]) @ modes[:, :rank]
        pca_train = ((train_x - mean) @ vectors[:, :rank]) / rms[:rank]
        pca_test = ((test_x - mean) @ vectors[:, :rank]) / rms[:rank]
        semantics = {
            "operator": feature_metrics(operator_train, operator_test, torch.from_numpy(discovery_labels).cuda(), torch.from_numpy(evaluation_labels).cuda(), evaluation_current, eligible, rank),
            "state_pca": feature_metrics(pca_train, pca_test, torch.from_numpy(discovery_labels).cuda(), torch.from_numpy(evaluation_labels).cuda(), evaluation_current, eligible, rank),
        }
        rotation_singles = []
        for _ in range(config["random_rotations"]):
            rotation = torch.linalg.qr(
                torch.randn(rank, rank, generator=random_generator, device="cuda")
            ).Q
            rotated = feature_metrics(
                operator_train @ rotation, operator_test @ rotation,
                torch.from_numpy(discovery_labels).cuda(), torch.from_numpy(evaluation_labels).cuda(),
                evaluation_current, eligible, rank, include_within=False,
            )
            rotation_singles.append(rotated["single_median"])
        semantics["rotation_single_p95"] = float(np.quantile(rotation_singles, .95))

        energy = singular.square()
        decoded = vectors @ (rms[:, None] * modes[:, :rank])
        decoded_directions[layer] = torch.linalg.qr(decoded, mode="reduced").Q
        results[str(layer)] = {
            "probe_top64_overlap": subspace_overlap(modes[:, :rank], modes_2[:, :rank]),
            "operator_top64_energy": float(energy[:rank].sum() / energy.sum()),
            "operator_effective_rank": float(energy.sum().square() / energy.square().sum()),
            "rank": rank_rows,
            "random_rank64_r2_p95": float(np.quantile(random_r2, .95)),
            "semantics": semantics,
        }
        bases[str(layer)] = {
            "mean": mean.cpu(), "state_vectors": vectors.cpu(), "state_rms": rms.cpu(),
            "operator_modes": modes.cpu(), "operator_singular": singular.cpu(),
        }
        del train_x, test_x, first, second, exact_sketch

    adjacent = {}
    for first, second in zip(config["layers"][:-1], config["layers"][1:]):
        adjacent[f"{first}-{second}"] = subspace_overlap(decoded_directions[first], decoded_directions[second])

    hypotheses = {
        "O0_probe_replication_identifies_operator_modes": {"supported": all(row["probe_top64_overlap"] >= .80 for row in results.values())},
        "O1_rank64_operator_vocabulary_is_predictive": {"supported": all(row["rank"]["64"]["operator_r2"] >= .50 for row in results.values())},
        "O2_operator_basis_beats_state_pca": {"supported": all(row["rank"]["64"]["operator_r2"] - row["rank"]["64"]["state_pca_r2"] >= .05 for row in results.values())},
        "O3_rank256_is_high_fidelity": {"supported": all(row["rank"]["256"]["operator_r2"] >= .90 for row in results.values())},
        "O4_operator_axes_are_semantically_privileged": {"supported": sum(row["semantics"]["operator"]["single_median"] - row["semantics"]["rotation_single_p95"] >= .02 for row in results.values()) >= 4},
        "O5_operator_subspace_improves_context_semantics": {"supported": sum(row["semantics"]["operator"]["within_token_median"] - row["semantics"]["state_pca"]["within_token_median"] >= .02 for row in results.values()) >= 4},
        "O6_operator_directions_recur_across_depth": {"supported": all(value >= .15 for value in adjacent.values())},
    }
    payload = {"config": config, "maximum_tensor_identity_error": maximum_identity_error, "hypotheses": hypotheses, "results": results, "adjacent_overlap": adjacent}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save(bases, BASES)

    lines = [
        "# Weight-derived early operator-tensor vocabulary", "",
        "For each bilinear MLP, random Kronecker probes sketch the exact linear map from the current normalized state to the complete local MLP Jacobian. Natural-state covariance whitens the state coordinates; an SVD then supplies a label-free operator-aware basis. A second independent weight sketch and disjoint FineWeb states evaluate reconstruction. State PCA, random whitened subspaces, and rotations within the same operator subspace are controls.", "",
        f"Maximum explicit-versus-implicit tensor identity error: `{maximum_identity_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Operator vocabulary](bilin18_early_operator_tensor_vocabulary.png)", "", "| Layer | Probe overlap | Eff. rank | Top64 energy | R2 op64 / PCA64 / random p95 | R2 op256 | Semantic one / rotation p95 | Within-token op/PCA |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; semantic = row["semantics"]
        lines.append(f"| {layer} | {row['probe_top64_overlap']:.3f} | {row['operator_effective_rank']:.1f} | {row['operator_top64_energy']:.3f} | {row['rank']['64']['operator_r2']:.3f} / {row['rank']['64']['state_pca_r2']:.3f} / {row['random_rank64_r2_p95']:.3f} | {row['rank']['256']['operator_r2']:.3f} | {semantic['operator']['single_median']:.3f} / {semantic['rotation_single_p95']:.3f} | {semantic['operator']['within_token_median']:.3f}/{semantic['state_pca']['within_token_median']:.3f} |")
    lines += ["", "Adjacent-layer overlap of the decoded top-64 state-direction subspaces: `" + json.dumps(adjacent, sort_keys=True) + "`.", "", "## Interpretation boundary", "", "These modes are canonical for Frobenius prediction of the local MLP Jacobian under the measured FineWeb state covariance. They are not automatically semantic concepts or complete transformer circuits. Kronecker sketches are unbiased but finite; the independent-sketch overlap and reconstruction columns expose sketch error. A positive operator reconstruction result still requires suffix composition and finite causal validation before it becomes a circuit explanation."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    layers = config["layers"]
    for rank in config["ranks"]:
        axes[0].plot(layers, [results[str(layer)]["rank"][str(rank)]["operator_r2"] for layer in layers], marker="o", label=f"rank {rank}")
    axes[0].set(xlabel="Layer", ylabel="Independent-sketch centered R²", title="Operator-aware reconstruction"); axes[0].grid(alpha=.2); axes[0].legend(ncol=2)
    axes[1].plot(layers, [results[str(layer)]["rank"]["64"]["operator_r2"] for layer in layers], marker="o", label="operator")
    axes[1].plot(layers, [results[str(layer)]["rank"]["64"]["state_pca_r2"] for layer in layers], marker="o", label="state PCA")
    axes[1].plot(layers, [results[str(layer)]["random_rank64_r2_p95"] for layer in layers], marker="o", label="random p95")
    axes[1].set(xlabel="Layer", ylabel="Rank-64 centered R²", title="Does tensor geometry select better states?"); axes[1].grid(alpha=.2); axes[1].legend()
    axes[2].plot(layers, [results[str(layer)]["semantics"]["operator"]["single_median"] for layer in layers], marker="o", label="operator axis")
    axes[2].plot(layers, [results[str(layer)]["semantics"]["rotation_single_p95"] for layer in layers], marker="o", label="rotation p95")
    axes[2].plot(layers, [results[str(layer)]["semantics"]["operator"]["within_token_median"] for layer in layers], marker="o", label="operator 64 within-token")
    axes[2].set(xlabel="Layer", ylabel="Median held-out AUROC", title="Semantic red team"); axes[2].grid(alpha=.2); axes[2].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_tensor_identity_error": maximum_identity_error, "results": results, "adjacent_overlap": adjacent}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
