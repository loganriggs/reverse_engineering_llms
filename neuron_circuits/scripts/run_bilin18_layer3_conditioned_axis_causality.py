#!/usr/bin/env python3
"""Held-out causal validation of individual conditioned spectral axes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_conditioned_axis_causality.json")
OUTPUT = TENSOR_ROOT / "layer3_conditioned_axis_causality.json"
REPORT = TENSOR_ROOT / "LAYER3_CONDITIONED_AXIS_CAUSALITY.md"
FIGURE = TENSOR_ROOT / "layer3_conditioned_axis_causality.png"


def spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values))
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def pair_balanced(records: list[dict], getter) -> float:
    groups = defaultdict(list)
    for record in records:
        groups[f"{record['source_id']}->{record['target_id']}"].append(float(getter(record)))
    return median([median(values) for values in groups.values()])


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    spike_source = json.loads(Path(config["source_results"]).read_text())
    spike_arrays = np.load(config["source_arrays"])
    control_source = json.loads(Path(config["control_results"]).read_text())
    control_arrays = np.load(config["control_arrays"])
    unseen_source = json.loads(Path(config["unseen_results"]).read_text())
    quotient_config = json.loads(Path(config["source_config"]).read_text())
    semantic_config = json.loads(Path(quotient_config["source_config"]).read_text())
    families = list(spike_source["results"])
    corpora = ("fineweb", "tinystories")

    # Preserve pair balance by taking at most the first two frozen natural contexts per pair.
    counts = defaultdict(int)
    panel = []
    for record in unseen_source["records"]:
        key = (record["corpus"], record["family"], record["source_id"], record["target_id"])
        if counts[key] < config["contexts_per_pair"]:
            panel.append(record)
            counts[key] += 1

    arrays = {
        corpus: np.asarray(
            np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]],
            dtype=np.int64,
        )
        for corpus, spec in semantic_config["corpora"].items() if corpus in corpora
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer = config["layer"]
    block = model.blocks[layer]

    canonical_basis = {
        family: torch.from_numpy(spike_arrays[f"{family}_canonical_basis"]).cuda()
        for family in families
    }
    systems = {}
    for family in families:
        systems[family] = []
        for axis in range(canonical_basis[family].shape[1]):
            direction = canonical_basis[family][:, axis]
            coefficients = block.mlp.Down.weight.float().T @ direction
            matrix = block.mlp.Left.weight.float().T @ (
                coefficients[:, None] * block.mlp.Right.weight.float()
            )
            matrix = 0.5 * (matrix + matrix.T)
            values, vectors = torch.linalg.eigh(matrix)
            order = values.abs().argsort(descending=True)
            systems[family].append((values[order], vectors[:, order]))

    random_order = np.argsort([
        row["median_top4"] for row in control_source["random_subspaces"]
    ])[::-1][:config["random_control_subspaces"]]
    random_control = [
        torch.from_numpy(control_arrays[f"random_subspace_{index}"]).cuda()
        for index in random_order
    ]

    evaluated = []
    maximum_error = 0.0
    for record in panel:
        raw = arrays[record["corpus"]][
            record["row"],
            record["source_start"]:record["source_start"] + config["sequence_length"],
        ].copy()
        changed = raw.copy()
        changed[config["edit_position"]] = record["target_id"]
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer,
        )[layer]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )[layer]
        state = channel_state(model, layer, base, counterfactual, config["causal_position"])
        target = state["target_write"].float()
        family = record["family"]
        basis = canonical_basis[family]
        exact_axes = [
            (target @ basis[:, axis:axis + 1]) * basis[:, axis]
            for axis in range(basis.shape[1])
        ]
        family_exact = sum(exact_axes)
        exact_recomposition_error = float(
            (family_exact - (target @ basis) @ basis.T).norm()
            / family_exact.norm().clamp_min(1e-30)
        )
        maximum_error = max(maximum_error, exact_recomposition_error)
        writes = {
            "full_mlp": target,
            "family_exact": family_exact,
            "cumulative_1": exact_axes[0],
            "cumulative_2": exact_axes[0] + exact_axes[1],
        }
        for axis, exact in enumerate(exact_axes):
            writes[f"axis_{axis}"] = exact
            writes[f"omit_axis_{axis}"] = family_exact - exact

        dimension = base["pre"].shape[-1]
        xb = F.rms_norm(base["pre"], (dimension,)).float()
        xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
        for rank in config["spectral_ranks"]:
            predicted = torch.zeros_like(target)
            for axis, (values, vectors) in enumerate(systems[family]):
                base_coordinates = xb @ vectors[:, :rank]
                changed_coordinates = xc @ vectors[:, :rank]
                scalar = (
                    (changed_coordinates.square() - base_coordinates.square())
                    * values[:rank]
                ).sum(-1, keepdim=True)
                predicted += scalar * basis[:, axis]
            writes[f"spectral_{rank}"] = predicted

        wrong_family = []
        for wrong in families:
            if wrong == family:
                continue
            name = f"wrong_family:{wrong}"
            wrong_basis = canonical_basis[wrong]
            writes[name] = (target @ wrong_basis) @ wrong_basis.T
            wrong_family.append(name)
        random_names = []
        for index, random_basis in enumerate(random_control):
            name = f"random_control:{index}"
            writes[name] = (target @ random_basis) @ random_basis.T
            random_names.append(name)
        effects = evaluate_writes(
            model, layer, base, counterfactual, writes, config["causal_position"],
        )
        best_wrong = max(effects[name]["family_patch_fidelity"] for name in wrong_family)
        best_random = max(effects[name]["family_patch_fidelity"] for name in random_names)
        evaluated.append({
            **{key: record[key] for key in (
                "corpus", "family", "source_id", "target_id", "row", "source_start",
            )},
            "effects": effects,
            "best_wrong_family_fidelity": best_wrong,
            "best_random_control_fidelity": best_random,
            "exact_recomposition_error": exact_recomposition_error,
        })
        print("axis causal", record["corpus"], family, record["source_id"], flush=True)

    summary = {corpus: {} for corpus in corpora}
    for corpus in corpora:
        for family in families:
            selected = [
                record for record in evaluated
                if record["corpus"] == corpus and record["family"] == family
            ]
            row = {
                "contexts": len(selected),
                "pairs": len({(r["source_id"], r["target_id"]) for r in selected}),
                "family_full_mlp_fidelity": pair_balanced(
                    selected, lambda r: r["effects"]["family_exact"]["full_mlp_fidelity"],
                ),
                "family_semantic_projection": pair_balanced(
                    selected, lambda r: r["effects"]["family_exact"]["semantic_target_projection"],
                ),
                "cumulative_1": pair_balanced(
                    selected, lambda r: r["effects"]["cumulative_1"]["family_patch_fidelity"],
                ),
                "cumulative_2": pair_balanced(
                    selected, lambda r: r["effects"]["cumulative_2"]["family_patch_fidelity"],
                ),
                "best_wrong": pair_balanced(selected, lambda r: r["best_wrong_family_fidelity"]),
                "best_random": pair_balanced(selected, lambda r: r["best_random_control_fidelity"]),
            }
            for rank in config["spectral_ranks"]:
                row[f"spectral_{rank}"] = pair_balanced(
                    selected, lambda r, rank=rank:
                    r["effects"][f"spectral_{rank}"]["family_patch_fidelity"],
                )
            for axis in range(4):
                row[f"axis_{axis}_sufficiency"] = pair_balanced(
                    selected, lambda r, axis=axis:
                    r["effects"][f"axis_{axis}"]["family_patch_fidelity"],
                )
                row[f"axis_{axis}_necessity"] = pair_balanced(
                    selected, lambda r, axis=axis:
                    1 - r["effects"][f"omit_axis_{axis}"]["family_patch_fidelity"],
                )
            summary[corpus][family] = row

    spectral_concentration = []
    axis_necessity = []
    for family in families:
        for axis in range(4):
            spectral_concentration.append(
                spike_source["results"][family]["axes"][axis]["observed"]["top4_energy_fraction"]
            )
            axis_necessity.append(median([
                abs(summary[corpus][family][f"axis_{axis}_necessity"])
                for corpus in corpora
            ]))
    causal_spearman = spearman(spectral_concentration, axis_necessity)

    hypotheses = {
        "A1_canonical_axes_exactly_recompose_family_patch": {
            "supported": maximum_error <= 1e-5,
            "maximum_error": maximum_error,
        },
        "A2_leading_axis_is_often_sufficient": {
            "supported": sum(all(
                summary[corpus][family]["cumulative_1"] >= .50 for corpus in corpora
            ) for family in families) >= 3,
        },
        "A3_two_axes_are_often_sufficient": {
            "supported": sum(all(
                summary[corpus][family]["cumulative_2"] >= .70 for corpus in corpora
            ) for family in families) >= 3,
        },
        "A4_family_subspaces_beat_exceptional_controls": {
            "supported": all(
                1 - summary[corpus][family][control] >= .10
                for corpus in corpora for family in families
                for control in ("best_wrong", "best_random")
            ),
        },
        "A5_rank64_modes_reproduce_unseen_family_effects": {
            "supported": all(
                summary[corpus][family]["spectral_64"] >= .90
                for corpus in corpora for family in families
            ),
        },
        "A6_spectral_concentration_predicts_causal_importance": {
            "supported": causal_spearman >= .50,
            "spearman": causal_spearman,
        },
        "A7_exceptional_axes_are_individually_necessary": {
            "supported": sum(all(
                summary[corpus][family]["axis_0_necessity"] >= .20 for corpus in corpora
            ) for family in families) >= 3,
        },
        "A8_family_patch_remains_partial": {
            "supported": all(
                summary[corpus][family]["family_full_mlp_fidelity"] < .30
                for corpus in corpora for family in families
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summary": summary,
        "spectral_concentration": spectral_concentration,
        "axis_necessity": axis_necessity,
        "records": evaluated,
    }, indent=2))

    figure, axes_plot = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(families))
    width = .18
    for index, (corpus, metric) in enumerate((
        ("fineweb", "cumulative_1"), ("fineweb", "cumulative_2"),
        ("tinystories", "cumulative_1"), ("tinystories", "cumulative_2"),
    )):
        axes_plot[0].bar(
            x + (index - 1.5) * width,
            [summary[corpus][family][metric] for family in families],
            width, label=f"{corpus} {metric}",
        )
    axes_plot[0].axhline(1, color="black", linewidth=1)
    axes_plot[0].set_xticks(x, families, rotation=25, ha="right")
    axes_plot[0].set_ylabel("Fidelity to exact family output-4 patch")
    axes_plot[0].set_title("Tensor-energy ordering: sufficiency")
    axes_plot[0].legend(fontsize=7)
    axes_plot[1].scatter(spectral_concentration, axis_necessity)
    offsets = {
        "sentiment": (8, -12), "gender": (-8, 10), "tense": (8, 12),
        "increasing_scale": (8, -20), "polarity": (8, 4),
    }
    for index, family in enumerate(families):
        axes_plot[1].annotate(
            family,
            (spectral_concentration[4 * index], axis_necessity[4 * index]),
            xytext=offsets[family], textcoords="offset points", fontsize=8,
            arrowprops={"arrowstyle": "-", "linewidth": .5, "color": "0.4"},
        )
    axes_plot[1].set_xlabel("Axis top-4 interaction energy")
    axes_plot[1].set_ylabel("Held-out absolute axis necessity")
    axes_plot[1].set_title(f"Spectral vs causal rank (Spearman {causal_spearman:.2f})")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Held-out causality of Layer-3 conditioned spectral axes",
        "",
        "Canonical axes and input eigensystems are frozen from the weight analysis. They are evaluated on disjoint unseen lexical substitutions, with at most two natural contexts per pair and pair-balanced aggregation. Wrong semantic families and the five most spectrally concentrated canonicalized random output subspaces are controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Conditioned axis causality](layer3_conditioned_axis_causality.png)",
        "",
        "| Corpus / family | Pairs | Axis 0 | Axes 0:2 | Spectral-16 | Spectral-64 | Best wrong | Best random | Full-MLP fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        for family in families:
            row = summary[corpus][family]
            lines.append(
                f"| {corpus} / {family} | {row['pairs']} | {row['cumulative_1']:.3f} | "
                f"{row['cumulative_2']:.3f} | {row['spectral_16']:.3f} | "
                f"{row['spectral_64']:.3f} | {row['best_wrong']:.3f} | "
                f"{row['best_random']:.3f} | {row['family_full_mlp_fidelity']:.3f} |"
            )
    lines += [
        "",
        f"Across the 20 axes, top-four spectral concentration versus held-out absolute necessity has Spearman **{causal_spearman:.3f}**.",
        "",
        "Fidelity is measured relative to the exact output-4 family patch. Full-MLP fidelity is shown separately. An axis can faithfully reproduce or be necessary for that restricted patch while the parent patch remains a small part of the entire counterfactual computation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
