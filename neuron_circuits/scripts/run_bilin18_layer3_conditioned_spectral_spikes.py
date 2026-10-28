#!/usr/bin/env python3
"""Test Thomas-Dooms-style spikes after semantic output conditioning at Layer 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw
from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_layer3_family_interaction_modes import output_basis
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_conditioned_spectral_spikes.json")
OUTPUT = TENSOR_ROOT / "layer3_conditioned_spectral_spikes.json"
ARRAYS = TENSOR_ROOT / "layer3_conditioned_spectral_spikes.npz"
REPORT = TENSOR_ROOT / "LAYER3_CONDITIONED_SPECTRAL_SPIKES.md"
FIGURE = TENSOR_ROOT / "layer3_conditioned_spectral_spikes.png"


def interaction_matrix(left: torch.Tensor, right: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    matrix = left.T @ (coefficients[:, None] * right)
    return 0.5 * (matrix + matrix.T)


def interaction_spectrum(left: torch.Tensor, right: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.eigvalsh(interaction_matrix(left, right, coefficients))
    return values.abs().sort(descending=True).values


def summarize_spectrum(values: torch.Tensor, config: dict) -> dict:
    fit = fit_powerlaw(
        values.cpu().numpy(), config["bulk_rank_start"], config["bulk_rank_stop"],
        config["outlier_count"],
    )
    return fit | {"leading": float(values[0])}


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    family_config = json.loads(Path(config["source_config"]).read_text())
    lopo_config = json.loads(Path(family_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    native_saved = json.loads(Path(config["native_results"]).read_text())
    global_saved = json.loads(Path(config["global_spectrum_results"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    families = list(parent["families"])
    layer = config["layer"]
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[layer]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    channel_gram = symmetric_channel_gram(left, right)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }

    def reconstruct(record: dict) -> tuple[dict, dict]:
        spec = fixed["evaluation"][record["corpus"]]
        row = arrays[record["corpus"]][spec["start"] + record["row"]]
        raw = np.asarray(
            row[record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)[layer]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )[layer]
        return base, counterfactual

    discovery = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    writes = {family: [] for family in families}
    for record in discovery:
        base, counterfactual = reconstruct(record)
        state = channel_state(model, layer, base, counterfactual, fixed["causal_position"])
        writes[record["family"]].append(
            state["target_write"][0, fixed["edit_position"]:fixed["causal_position"] + 1]
        )
        print("discovery", record["family"], record["pair"], flush=True)

    random_output_rows = []
    for replicate in range(config["random_output_directions"]):
        direction = torch.randn(down.shape[0], generator=generator, device="cuda")
        direction /= direction.norm()
        values = interaction_spectrum(left, right, down.T @ direction)
        row = summarize_spectrum(values, config)
        random_output_rows.append(row)
        print("random_output", replicate, flush=True)
    random_top4 = np.asarray([row["top4_energy_fraction"] for row in random_output_rows])
    random_top4_95 = float(np.quantile(random_top4, .95))

    results = {}
    saved_arrays = {"random_top4_energy": random_top4}
    maximum_projector_error = 0.0
    maximum_energy_error = 0.0
    for family in families:
        usage_basis = output_basis(writes[family], config["output_rank"])
        coefficients = down.T @ usage_basis
        restricted_gram = coefficients.T @ (channel_gram @ coefficients)
        restricted_gram = 0.5 * (restricted_gram + restricted_gram.T)
        energy_values, rotation = torch.linalg.eigh(restricted_gram)
        order = energy_values.argsort(descending=True)
        energy_values, rotation = energy_values[order], rotation[:, order]
        canonical = usage_basis @ rotation
        canonical_coefficients = coefficients @ rotation
        saved_arrays[f"{family}_canonical_basis"] = canonical.cpu().numpy()
        saved_arrays[f"{family}_canonical_coefficients"] = canonical_coefficients.cpu().numpy()
        projector_error = float(
            (canonical @ canonical.T - usage_basis @ usage_basis.T).norm()
            / (usage_basis @ usage_basis.T).norm().clamp_min(1e-30)
        )
        reconstructed_gram = rotation.T @ restricted_gram @ rotation
        energy_error = float(
            (reconstructed_gram.diag() - energy_values).norm()
            / energy_values.norm().clamp_min(1e-30)
        )
        maximum_projector_error = max(maximum_projector_error, projector_error)
        maximum_energy_error = max(maximum_energy_error, energy_error)
        axes = []
        for axis in range(config["output_rank"]):
            values = interaction_spectrum(left, right, canonical_coefficients[:, axis])
            observed = summarize_spectrum(values, config)
            nulls = []
            for replicate in range(config["null_replicates"]):
                signs = torch.randint(
                    0, 2, (down.shape[1],), generator=generator, device="cuda",
                ) * 2 - 1
                sign_null = interaction_spectrum(
                    left, right, canonical_coefficients[:, axis] * signs,
                )
                nulls.append(summarize_spectrum(sign_null, config))
            null_max = max(row["leading"] for row in nulls)
            row = {
                "axis": axis,
                "restricted_tensor_energy": float(energy_values[axis]),
                "observed": observed,
                "sign_nulls": nulls,
                "leading_to_null_max": observed["leading"] / max(null_max, 1e-30),
                "beats_every_sign_null": observed["leading"] > null_max,
                "beats_random_top4_95": observed["top4_energy_fraction"] > random_top4_95,
            }
            axes.append(row)
            saved_arrays[f"{family}_axis{axis}_spectrum"] = values.cpu().numpy()
            print("family", family, "axis", axis, flush=True)
        results[family] = {
            "projector_error": projector_error,
            "restricted_energy_error": energy_error,
            "axes": axes,
            "median_top4_energy": float(np.median([
                row["observed"]["top4_energy_fraction"] for row in axes
            ])),
            "median_leading_to_null_max": float(np.median([
                row["leading_to_null_max"] for row in axes
            ])),
            "axes_beating_all_nulls": sum(row["beats_every_sign_null"] for row in axes),
            "axes_with_strict_powerlaw_spike": sum(
                row["observed"]["top_log_excess"][0] > 0 for row in axes
            ),
        }

    global_top4 = float(global_saved["empirical_null"]["observed_top4_energy"])
    hypotheses = {
        "P1_weight_canonicalization_is_exact": {
            "supported": maximum_projector_error <= 1e-5 and maximum_energy_error <= 1e-5,
            "maximum_projector_error": maximum_projector_error,
            "maximum_energy_error": maximum_energy_error,
        },
        "P2_conditioned_axes_are_coordination_outliers": {
            "supported": sum(
                results[family]["median_leading_to_null_max"] > 1 for family in families
            ) >= 3,
        },
        "P3_conditioning_enriches_low_rank_forms": {
            "supported": sum(
                results[family]["median_top4_energy"] > random_top4_95 for family in families
            ) >= 3,
            "random_output_top4_95": random_top4_95,
        },
        "P4_conditioned_powerlaw_spikes_are_common": {
            "supported": sum(
                results[family]["axes_with_strict_powerlaw_spike"] >= 1 for family in families
            ) >= 3,
        },
        "P5_every_family_has_an_exceptional_axis": {
            "supported": all(results[family]["axes_beating_all_nulls"] >= 1 for family in families),
        },
        "P6_family_conditioning_beats_global_concentration": {
            "supported": all(results[family]["median_top4_energy"] > global_top4 for family in families),
            "global_top4_energy": global_top4,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "random_output_controls": random_output_rows,
        "results": results,
    }, indent=2))
    np.savez_compressed(ARRAYS, **saved_arrays)

    figure, axes_plot = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(families))
    axes_plot[0].bar(x, [results[f]["median_top4_energy"] for f in families])
    axes_plot[0].axhline(random_top4_95, color="tab:red", linestyle="--", label="random output 95th")
    axes_plot[0].axhline(global_top4, color="black", linestyle=":", label="global tensor top-4")
    axes_plot[0].set_xticks(x, families, rotation=25, ha="right")
    axes_plot[0].set_ylabel("Median top-4 interaction energy")
    axes_plot[0].set_title("Low-rank enrichment after conditioning")
    axes_plot[0].legend()
    axes_plot[1].bar(x, [results[f]["median_leading_to_null_max"] for f in families])
    axes_plot[1].axhline(1, color="black", linewidth=1)
    axes_plot[1].set_xticks(x, families, rotation=25, ha="right")
    axes_plot[1].set_ylabel("Median leading eigenvalue / max sign null")
    axes_plot[1].set_title("Coordination outlier strength")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Output-conditioned spectral spikes in Layer 3",
        "",
        "Each frozen four-dimensional semantic-family output subspace is rotated into canonical directions by diagonalizing the exact tensor Frobenius Gram restricted to that subspace. This rotation uses weights after the family subspace has been frozen and cannot change the subspace. Each resulting output direction contracts the bilinear tensor into an exact symmetric input interaction matrix. Its spectrum is compared with channel-sign nulls and isotropic random output directions.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Conditioned spectral spikes](layer3_conditioned_spectral_spikes.png)",
        "",
        "| Family | Median top-4 energy | Median leading/null-max | Axes beating all nulls | Axes with strict fitted spike |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in families:
        row = results[family]
        lines.append(
            f"| {family} | {row['median_top4_energy']:.3f} | "
            f"{row['median_leading_to_null_max']:.3f} | {row['axes_beating_all_nulls']}/4 | "
            f"{row['axes_with_strict_powerlaw_spike']}/4 |"
        )
    lines += [
        "",
        f"The random-output 95th percentile top-four energy is **{random_top4_95:.3f}**; "
        f"the global Layer-3 tensor-map top-four energy is **{global_top4:.3f}**.",
        "",
        "A null-exceeding conditioned axis is an exceptional interaction form relative to the tested coordination null. It is not automatically a complete semantic circuit: the family output subspace was discovered from data, and prior finite interventions show that these rank-four output projections explain only a minority of the complete MLP effect. Conversely, failure of a fitted positive power-law excess does not erase causal semantic structure already validated in the same family.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
