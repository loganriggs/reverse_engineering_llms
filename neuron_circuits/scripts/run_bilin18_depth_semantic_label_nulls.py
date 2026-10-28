#!/usr/bin/env python3
"""Matched semantic-label nulls built from the same natural counterfactual writes."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw
from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_layer3_conditioned_spectral_spikes import interaction_spectrum
from run_bilin18_layer3_family_interaction_modes import output_basis
from run_bilin18_layer3_frozen_channel_transfer import recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_semantic_label_nulls.json")
OUTPUT = TENSOR_ROOT / "depth_semantic_label_nulls.json"
REPORT = TENSOR_ROOT / "DEPTH_SEMANTIC_LABEL_NULLS.md"
FIGURE = TENSOR_ROOT / "depth_semantic_label_nulls.png"


def top4_fraction(values: torch.Tensor) -> float:
    energy = values.square()
    return float(energy[:4].sum() / energy.sum().clamp_min(1e-30))


def overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.T @ right).square().sum() / min(left.shape[1], right.shape[1]))


def empirical_p(observed: float, nulls: list[float]) -> float:
    return (1 + sum(value >= observed for value in nulls)) / (len(nulls) + 1)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    family_config = json.loads(Path(config["source_config"]).read_text())
    lopo_config = json.loads(Path(family_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    native = json.loads(Path(config["native_results"]).read_text())
    depth = json.loads(Path(config["depth_results"]).read_text())
    families = list(parent["families"])
    layers = config["layers"]
    rng = np.random.default_rng(config["seed"])
    discovery = [
        record for record in native["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    group_sizes = [sum(record["family"] == family for record in discovery) for family in families]
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    pair_writes = {layer: [] for layer in layers}
    labels = []
    for record in discovery:
        spec = fixed["evaluation"][record["corpus"]]
        row = arrays[record["corpus"]][spec["start"] + record["row"]]
        raw = np.asarray(
            row[record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=max(layers))
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=max(layers),
        )
        for layer in layers:
            delta = counterfactual[layer]["mlp_write"].float() - base[layer]["mlp_write"].float()
            pair_writes[layer].append(delta[
                0, fixed["edit_position"]:fixed["causal_position"] + 1
            ])
        labels.append(record["family"])
        print("discovery", record["family"], record["pair"], flush=True)

    results = {}
    for layer in layers:
        print("layer", layer, flush=True)
        block = model.blocks[layer]
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        down = block.mlp.Down.weight.float()
        channel_gram = symmetric_channel_gram(left, right)
        label_shuffle = []
        for replicate in range(config["label_shuffle_replicates"]):
            permutation = rng.permutation(len(pair_writes[layer]))
            cursor = 0
            group_bases = []
            group_median_top4 = []
            strict_axes = 0
            for size in group_sizes:
                indices = permutation[cursor:cursor + size]
                cursor += size
                basis = output_basis([pair_writes[layer][index] for index in indices], config["output_rank"])
                coefficients = down.T @ basis
                gram = coefficients.T @ (channel_gram @ coefficients)
                gram = 0.5 * (gram + gram.T)
                _, rotation = torch.linalg.eigh(gram)
                canonical = basis @ rotation.flip(1)
                group_bases.append(canonical)
                fractions = []
                for axis in range(config["output_rank"]):
                    spectrum = interaction_spectrum(left, right, down.T @ canonical[:, axis])
                    fractions.append(top4_fraction(spectrum))
                    fit = fit_powerlaw(
                        spectrum.cpu().numpy(), config["bulk_rank_start"],
                        config["bulk_rank_stop"], config["outlier_count"],
                    )
                    strict_axes += fit["top_log_excess"][0] > 0
                group_median_top4.append(float(np.median(fractions)))
            pairwise_overlap = [
                overlap(group_bases[left_index], group_bases[right_index])
                for left_index, right_index in combinations(range(len(group_bases)), 2)
            ]
            label_shuffle.append({
                "median_top4_energy": float(np.median(group_median_top4)),
                "median_pairwise_overlap": float(np.median(pairwise_overlap)),
                "strict_axes": int(strict_axes),
            })
            print(" label_shuffle", replicate, flush=True)
        observed = depth["results"][str(layer)]
        null_top4 = [row["median_top4_energy"] for row in label_shuffle]
        null_overlap = [row["median_pairwise_overlap"] for row in label_shuffle]
        null_strict = [row["strict_axes"] for row in label_shuffle]
        observed_top4 = observed["median_family_top4_energy"]
        null_mean = float(np.mean(null_top4))
        null_std = float(np.std(null_top4, ddof=1))
        results[str(layer)] = {
            "observed_median_top4": observed_top4,
            "label_null_mean_top4": null_mean,
            "label_null_95_top4": float(np.quantile(null_top4, .95)),
            "label_null_p": empirical_p(observed_top4, null_top4),
            "label_null_z": (observed_top4 - null_mean) / max(null_std, 1e-30),
            "observed_family_overlap": float(np.median([
                np.square(
                    np.asarray(
                        np.load("artifacts/bilin18_tensor_decomposition/depth_conditioned_semantic_spikes.npz")
                        [f"layer{layer}_{left_family}_basis"]
                    ).T
                    @ np.asarray(
                        np.load("artifacts/bilin18_tensor_decomposition/depth_conditioned_semantic_spikes.npz")
                        [f"layer{layer}_{right_family}_basis"]
                    )
                ).sum() / config["output_rank"]
                for left_family, right_family in combinations(families, 2)
            ])),
            "label_null_median_overlap": float(np.median(null_overlap)),
            "label_null_95_overlap": float(np.quantile(null_overlap, .95)),
            "observed_strict_axes": sum(
                observed["families"][family]["strict_axes"] for family in families
            ),
            "label_null_median_strict_axes": float(np.median(null_strict)),
            "isotropic_random_95": observed["random_subspace_median_top4_95"],
            "label_shuffle": label_shuffle,
        }
        del channel_gram

    early_z = max(results[str(layer)]["label_null_z"] for layer in (0, 3, 8))
    hypotheses = {
        "L1_early_semantic_grouping_enriches_conditioned_concentration": {
            "supported": sum(
                results[str(layer)]["observed_median_top4"]
                > results[str(layer)]["label_null_95_top4"]
                for layer in (0, 3, 8)
            ) >= 2,
        },
        "L2_late_conditioned_concentration_is_not_semantic_specific": {
            "supported": results["17"]["observed_median_top4"]
            <= results["17"]["label_null_95_top4"],
        },
        "L3_late_family_collapse_occurs_under_label_shuffle": {
            "supported": results["17"]["label_null_median_overlap"] >= .50
            and abs(
                results["17"]["label_null_median_overlap"]
                - results["17"]["observed_family_overlap"]
            ) <= .15,
        },
        "L4_late_strict_spikes_occur_under_label_shuffle": {
            "supported": results["17"]["label_null_median_strict_axes"] >= 15,
        },
        "L5_semantic_labels_matter_more_early_than_late": {
            "supported": early_z - results["17"]["label_null_z"] >= 1.0,
            "early_max_z": early_z,
            "late_z": results["17"]["label_null_z"],
        },
        "L6_isotropic_random_control_was_overoptimistic_late": {
            "supported": results["17"]["label_null_95_top4"]
            >= 5 * results["17"]["isotropic_random_95"],
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(layers))
    axes[0].bar(x, [results[str(layer)]["observed_median_top4"] for layer in layers], label="semantic labels")
    axes[0].plot(x, [results[str(layer)]["label_null_95_top4"] for layer in layers], "r--o", label="label-null 95th")
    axes[0].plot(x, [results[str(layer)]["isotropic_random_95"] for layer in layers], "k:o", label="isotropic 95th")
    axes[0].set_xticks(x, layers)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Median group top-4 energy")
    axes[0].set_title("Matched natural-write label null")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - .18, [results[str(layer)]["observed_family_overlap"] for layer in layers], .36, label="semantic labels")
    axes[1].bar(x + .18, [results[str(layer)]["label_null_median_overlap"] for layer in layers], .36, label="shuffled labels")
    axes[1].set_xticks(x, layers)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Median group-subspace overlap")
    axes[1].set_title("Late collapse is label-independent")
    axes[1].legend()
    axes[2].bar(x - .18, [results[str(layer)]["observed_strict_axes"] for layer in layers], .36, label="semantic labels")
    axes[2].bar(x + .18, [results[str(layer)]["label_null_median_strict_axes"] for layer in layers], .36, label="shuffled labels")
    axes[2].set_xticks(x, layers)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Strict fitted spikes (of 20)")
    axes[2].set_title("Power-law spikes under label shuffle")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Natural-write semantic-label nulls across depth",
        "",
        "Isotropic random output spaces do not match the data-dependent selection of semantic family spaces. This red-team preserves the exact 19 natural counterfactual write trajectories and group sizes, randomly reassigning lexical pairs to five pseudo-families before repeating output-basis fitting, tensor canonicalization, and input-spectrum measurement.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic-label nulls](depth_semantic_label_nulls.png)",
        "",
        "| Layer | Observed / label-null 95th top-4 | Empirical p / z | Observed / shuffled overlap | Strict observed / shuffled median | Isotropic 95th |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        row = results[str(layer)]
        lines.append(
            f"| {layer} | {row['observed_median_top4']:.3f} / {row['label_null_95_top4']:.3f} | "
            f"{row['label_null_p']:.3f} / {row['label_null_z']:.2f} | "
            f"{row['observed_family_overlap']:.3f} / {row['label_null_median_overlap']:.3f} | "
            f"{row['observed_strict_axes']} / {row['label_null_median_strict_axes']:.1f} | "
            f"{row['isotropic_random_95']:.3f} |"
        )
    lines += [
        "",
        "The label-shuffle null asks whether semantic grouping matters after conditioning on natural write selection. Passing only the isotropic control establishes that natural writes occupy an exceptional tensor region; exceeding the label null is required to attribute additional spectral concentration to the semantic partition itself.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
