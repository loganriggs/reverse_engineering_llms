#!/usr/bin/env python3
"""Replicate global versus semantic-conditioned tensor spectra across depth."""

from __future__ import annotations

import json
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


CONFIG = Path("configs/bilin18_depth_conditioned_semantic_spikes.json")
OUTPUT = TENSOR_ROOT / "depth_conditioned_semantic_spikes.json"
ARRAYS = TENSOR_ROOT / "depth_conditioned_semantic_spikes.npz"
REPORT = TENSOR_ROOT / "DEPTH_CONDITIONED_SEMANTIC_SPIKES.md"
FIGURE = TENSOR_ROOT / "depth_conditioned_semantic_spikes.png"


def top4_fraction(values: torch.Tensor) -> float:
    energy = values.square()
    return float(energy[:4].sum() / energy.sum().clamp_min(1e-30))


def empirical_p(observed: float, nulls: list[float]) -> float:
    return (1 + sum(value >= observed for value in nulls)) / (len(nulls) + 1)


def canonicalize(
    basis: torch.Tensor, down: torch.Tensor, channel_gram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    coefficients = down.T @ basis
    gram = coefficients.T @ (channel_gram @ coefficients)
    gram = 0.5 * (gram + gram.T)
    values, rotation = torch.linalg.eigh(gram)
    order = values.argsort(descending=True)
    canonical = basis @ rotation[:, order]
    projector_error = float(
        (canonical @ canonical.T - basis @ basis.T).norm()
        / (basis @ basis.T).norm().clamp_min(1e-30)
    )
    return canonical, values[order], projector_error


def normalized_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.square(left.T @ right).sum() / min(left.shape[1], right.shape[1]))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    family_config = json.loads(Path(config["source_config"]).read_text())
    lopo_config = json.loads(Path(family_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    native = json.loads(Path(config["native_results"]).read_text())
    families = list(parent["families"])
    layers = config["layers"]
    discovery = [
        record for record in native["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])

    writes = {layer: {family: [] for family in families} for layer in layers}
    for record in discovery:
        spec = fixed["evaluation"][record["corpus"]]
        row = arrays[record["corpus"]][spec["start"] + record["row"]]
        raw = np.asarray(
            row[record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=max(layers),
        )
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=max(layers),
        )
        for layer in layers:
            delta = counterfactual[layer]["mlp_write"].float() - base[layer]["mlp_write"].float()
            writes[layer][record["family"]].append(
                delta[0, fixed["edit_position"]:fixed["causal_position"] + 1]
            )
        print("discovery", record["family"], record["pair"], flush=True)

    results = {}
    bases = {}
    saved_arrays = {}
    for layer in layers:
        print("layer", layer, flush=True)
        block = model.blocks[layer]
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        down = block.mlp.Down.weight.float()
        channel_gram = symmetric_channel_gram(left, right)
        output_gram = down @ (channel_gram @ down.T)
        output_gram = 0.5 * (output_gram + output_gram.T)
        global_values, global_vectors = torch.linalg.eigh(output_gram)
        order = global_values.argsort(descending=True)
        global_values = global_values[order].clamp_min(0).sqrt()
        global_vectors = global_vectors[:, order]
        global_top4 = top4_fraction(global_values)
        global_fit = fit_powerlaw(
            global_values.cpu().numpy(), config["bulk_rank_start"], config["bulk_rank_stop"],
            config["outlier_count"],
        )

        random_subspace_rows = []
        for replicate in range(config["random_subspaces"]):
            raw_basis = torch.randn(
                down.shape[0], config["output_rank"], generator=generator, device="cuda",
            )
            random_subspace, _ = torch.linalg.qr(raw_basis, mode="reduced")
            canonical, _, _ = canonicalize(random_subspace, down, channel_gram)
            fractions = [
                top4_fraction(interaction_spectrum(left, right, down.T @ canonical[:, axis]))
                for axis in range(config["output_rank"])
            ]
            random_subspace_rows.append(float(np.median(fractions)))
            print(" random_subspace", replicate, flush=True)
        random_95 = float(np.quantile(random_subspace_rows, .95))

        family_rows = {}
        bases[layer] = {}
        for family in families:
            usage_basis = output_basis(writes[layer][family], config["output_rank"])
            canonical, tensor_energy, projector_error = canonicalize(
                usage_basis, down, channel_gram,
            )
            bases[layer][family] = canonical.cpu().numpy()
            saved_arrays[f"layer{layer}_{family}_basis"] = canonical.cpu().numpy()
            axes = []
            for axis in range(config["output_rank"]):
                coefficients = down.T @ canonical[:, axis]
                observed_values = interaction_spectrum(left, right, coefficients)
                observed_leading = float(observed_values[0])
                fit = fit_powerlaw(
                    observed_values.cpu().numpy(), config["bulk_rank_start"],
                    config["bulk_rank_stop"], config["outlier_count"],
                )
                sign_null = []
                pairing_null = []
                for replicate in range(config["null_replicates"]):
                    signs = torch.randint(
                        0, 2, (len(coefficients),), generator=generator, device="cuda",
                    ) * 2 - 1
                    sign_null.append(float(interaction_spectrum(
                        left, right, coefficients * signs,
                    )[0]))
                    permutation = torch.randperm(
                        len(coefficients), generator=generator, device="cuda",
                    )
                    pairing_null.append(float(interaction_spectrum(
                        left, right, coefficients[permutation],
                    )[0]))
                axes.append({
                    "axis": axis,
                    "tensor_energy": float(tensor_energy[axis]),
                    "top4_energy": top4_fraction(observed_values),
                    "leading_log_excess": fit["top_log_excess"][0],
                    "sign_p": empirical_p(observed_leading, sign_null),
                    "pairing_p": empirical_p(observed_leading, pairing_null),
                })
                print(" family", family, "axis", axis, flush=True)
            family_rows[family] = {
                "projector_error": projector_error,
                "write_norm": float(torch.cat(writes[layer][family]).norm()),
                "median_top4_energy": float(np.median([row["top4_energy"] for row in axes])),
                "beats_random_95": float(np.median([row["top4_energy"] for row in axes])) > random_95,
                "robust_both_axes": sum(
                    row["sign_p"] <= .05 and row["pairing_p"] <= .05 for row in axes
                ),
                "strict_axes": sum(row["leading_log_excess"] > 0 for row in axes),
                "global_top4_overlap": normalized_overlap(
                    canonical.cpu().numpy(), global_vectors[:, :4].cpu().numpy(),
                ),
                "axes": axes,
            }
        results[str(layer)] = {
            "global_top4_energy": global_top4,
            "global_fit": global_fit,
            "random_subspace_median_top4_95": random_95,
            "random_subspace_medians": random_subspace_rows,
            "families": family_rows,
            "median_family_top4_energy": float(np.median([
                family_rows[family]["median_top4_energy"] for family in families
            ])),
            "median_family_global_overlap": float(np.median([
                family_rows[family]["global_top4_overlap"] for family in families
            ])),
        }
        del channel_gram, output_gram

    recurrence = {}
    for left_layer, right_layer in zip(layers[:-1], layers[1:]):
        same_family_overlap = [
            normalized_overlap(bases[left_layer][family], bases[right_layer][family])
            for family in families
        ]
        wrong_family_overlap = [
            normalized_overlap(bases[left_layer][left_family], bases[right_layer][right_family])
            for left_family in families for right_family in families if left_family != right_family
        ]
        recurrence[f"{left_layer}->{right_layer}"] = {
            "same_family_overlap": same_family_overlap,
            "wrong_family_overlap": wrong_family_overlap,
            "same_mean": float(np.mean(same_family_overlap)),
            "wrong_mean": float(np.mean(wrong_family_overlap)),
            "gap": float(np.mean(same_family_overlap) - np.mean(wrong_family_overlap)),
        }

    prior = json.loads((TENSOR_ROOT / "layer3_conditioned_spectral_spikes.json").read_text())
    hypotheses = {
        "D1_canonicalization_is_exact_at_every_depth": {
            "supported": all(
                results[str(layer)]["families"][family]["projector_error"] <= 1e-5
                for layer in layers for family in families
            ),
        },
        "D2_conditioning_enriches_low_rank_forms_at_every_depth": {
            "supported": all(sum(
                results[str(layer)]["families"][family]["beats_random_95"] for family in families
            ) >= 3 for layer in layers),
        },
        "D3_conditioned_axes_survive_two_nulls_at_every_depth": {
            "supported": all(sum(
                results[str(layer)]["families"][family]["robust_both_axes"] >= 3
                for family in families
            ) >= 3 for layer in layers),
        },
        "D4_strict_powerlaw_spikes_have_low_recall_at_every_depth": {
            "supported": all(sum(
                results[str(layer)]["families"][family]["strict_axes"] for family in families
            ) <= 10 for layer in layers),
        },
        "D5_family_output_spaces_recur_across_depth": {
            "supported": all(row["gap"] >= .05 for row in recurrence.values()),
            "recurrence": recurrence,
        },
        "D6_global_spikes_align_more_with_families_late": {
            "supported": all(
                results["17"]["median_family_global_overlap"]
                - results[str(layer)]["median_family_global_overlap"] >= .05
                for layer in layers if layer != 17
            ),
        },
        "D7_late_global_concentration_implies_late_conditioned_concentration": {
            "supported": results["17"]["global_top4_energy"] == max(
                results[str(layer)]["global_top4_energy"] for layer in layers
            ) and results["17"]["median_family_top4_energy"] == max(
                results[str(layer)]["median_family_top4_energy"] for layer in layers
            ),
        },
        "D8_family_spaces_are_not_global_spike_spaces": {
            "supported": all(
                results[str(layer)]["median_family_global_overlap"] < .25 for layer in layers
            ),
        },
        "D9_layer3_replicates_prior_concentration": {
            "supported": all(abs(
                results["3"]["families"][family]["median_top4_energy"]
                - prior["results"][family]["median_top4_energy"]
            ) <= .01 for family in families),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
        "recurrence": recurrence,
    }, indent=2))
    np.savez_compressed(ARRAYS, **saved_arrays)

    figure, axes_plot = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(layers))
    axes_plot[0].bar(x - .18, [results[str(layer)]["global_top4_energy"] for layer in layers], .36, label="global")
    axes_plot[0].bar(x + .18, [results[str(layer)]["median_family_top4_energy"] for layer in layers], .36, label="conditioned family median")
    axes_plot[0].set_xticks(x, layers)
    axes_plot[0].set_xlabel("Layer")
    axes_plot[0].set_ylabel("Top-4 interaction energy")
    axes_plot[0].set_title("Global and conditioned concentration")
    axes_plot[0].legend()
    for family in families:
        axes_plot[1].plot(
            layers, [results[str(layer)]["families"][family]["median_top4_energy"] for layer in layers],
            marker="o", label=family,
        )
    axes_plot[1].set_xlabel("Layer")
    axes_plot[1].set_ylabel("Family median top-4 energy")
    axes_plot[1].set_title("Conditioned depth profiles")
    axes_plot[1].legend(fontsize=7)
    pairs = list(recurrence)
    axes_plot[2].bar(
        np.arange(len(pairs)) - .18, [recurrence[pair]["same_mean"] for pair in pairs], .36,
        label="same family",
    )
    axes_plot[2].bar(
        np.arange(len(pairs)) + .18, [recurrence[pair]["wrong_mean"] for pair in pairs], .36,
        label="wrong family",
    )
    axes_plot[2].set_xticks(np.arange(len(pairs)), pairs)
    axes_plot[2].set_ylabel("Normalized projector overlap")
    axes_plot[2].set_title("Cross-depth family recurrence")
    axes_plot[2].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Semantic-conditioned tensor spikes across depth",
        "",
        "The same frozen semantic edits define output-4 family spaces at layers 0, 3, 8, and 17. Each is rotated by its layer's exact tensor-Frobenius geometry. Conditioned spectra are compared with equally canonicalized random output-4 spaces plus sign and coefficient-pairing nulls. Global full-tensor spectra and cross-layer family-projector recurrence are reported separately.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Depth-conditioned semantic spikes](depth_conditioned_semantic_spikes.png)",
        "",
        "| Layer | Global top-4 | Conditioned median top-4 | Random 95th | Families above random | Median family/global overlap | Strict conditioned axes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        row = results[str(layer)]
        lines.append(
            f"| {layer} | {row['global_top4_energy']:.3f} | {row['median_family_top4_energy']:.3f} | "
            f"{row['random_subspace_median_top4_95']:.3f} | "
            f"{sum(row['families'][family]['beats_random_95'] for family in families)}/5 | "
            f"{row['median_family_global_overlap']:.3f} | "
            f"{sum(row['families'][family]['strict_axes'] for family in families)}/20 |"
        )
    lines += [
        "",
        "| Adjacent layers | Same-family overlap | Wrong-family overlap | Gap |",
        "|---|---:|---:|---:|",
    ]
    for pair, row in recurrence.items():
        lines.append(
            f"| {pair} | {row['same_mean']:.3f} | {row['wrong_mean']:.3f} | {row['gap']:.3f} |"
        )
    lines += [
        "",
        "Global concentration, conditioned concentration, and cross-depth recurrence answer different questions. A late global spike can be unrelated to the selected semantic families; a family can have low-rank input interactions without reusing the same residual-write subspace at another layer. The semantic subspaces are data-defined, while their internal canonicalization and spectra are weight-defined.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
