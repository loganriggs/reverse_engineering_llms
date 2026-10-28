#!/usr/bin/env python3
"""Generalize the context-conditioned tangent interpreter across components and corpora."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_query_conditioned_interpreter import ARRAYS as MAP_ARRAYS, CONFIG as INTERPRETER_CONFIG, collect_states_and_writes
from run_bilin18_query_map_tangent_interpreter import metrics
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_map_tangent_generalization.json")
OUTPUT = TENSOR_ROOT / "query_map_tangent_generalization.json"
REPORT = TENSOR_ROOT / "QUERY_MAP_TANGENT_GENERALIZATION.md"
FIGURE = TENSOR_ROOT / "query_map_tangent_generalization.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    saved = np.load(MAP_ARRAYS)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[17]
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    output_components = torch.from_numpy(saved["output_components"][:, : config["components"]]).cuda()
    singular = torch.from_numpy(saved["singular_values"][: config["components"]]).cuda()
    restricted = singular[:, None, None] * torch.from_numpy(saved["matrix_components"][: config["components"]]).cuda()
    semantic = torch.from_numpy(np.load(BASIS_ARRAYS)["semantic32"]).cuda()
    discovery_states, _, _ = collect_states_and_writes(
        model,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    discovery_x = torch.from_numpy(discovery_states).cuda()
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    state_v = state_vh.T
    bases = {"32": semantic}
    for dimension in (64, 128):
        bases[str(dimension)] = torch.linalg.qr(
            torch.cat((semantic, state_v[:, : dimension - 32]), dim=1), mode="reduced"
        ).Q

    inputs = []
    for component in range(config["components"]):
        values, vectors = torch.linalg.eigh(restricted[component])
        order = torch.argsort(values.abs(), descending=True)
        inputs.append(axes @ vectors[:, order[: config["eigenspace_rank"]]])

    corpus_data = {}
    for corpus, corpus_config in config["corpora"].items():
        states, writes, _ = collect_states_and_writes(
            model, corpus_config, interpreter_config["positions"], interpreter_config["batch_size"]
        )
        corpus_data[corpus] = {
            "states": torch.from_numpy(states).cuda(),
            "scores": (torch.from_numpy(writes).cuda() @ output_components).cpu().numpy(),
        }

    results = {}
    generator = np.random.default_rng(config["seed"])
    for corpus, data in corpus_data.items():
        x_all = data["states"]
        results[corpus] = {"high": {}, "random": {}}
        for component in range(config["components"]):
            coefficients = block.mlp.Down.weight.float().T @ output_components[:, component]
            matrix = block.mlp.Left.weight.float().T @ (
                coefficients[:, None] * block.mlp.Right.weight.float()
            )
            matrix = 0.5 * (matrix + matrix.T)
            centered = data["scores"][:, component] - np.median(data["scores"][:, component])
            high_indices = np.argpartition(np.abs(centered), -config["sites_per_stratum"])[-config["sites_per_stratum"] :]
            remaining = np.setdiff1d(np.arange(len(centered)), high_indices, assume_unique=False)
            random_indices = generator.choice(remaining, size=config["sites_per_stratum"], replace=False)
            for stratum, indices in (("high", high_indices), ("random", random_indices)):
                x = x_all[torch.as_tensor(indices, device="cuda")]
                vectors = inputs[component]
                delta = -(x @ vectors) @ vectors.T
                delta *= (
                    config["edit_norm_fraction"] * x.norm(dim=1, keepdim=True)
                    / delta.norm(dim=1, keepdim=True).clamp_min(1e-20)
                )
                actual = 2 * torch.einsum("ni,ij,nj->n", delta, matrix, x) + torch.einsum("ni,ij,nj->n", delta, matrix, delta)
                component_result = {}
                for name, basis in bases.items():
                    approximate_x = (x @ basis) @ basis.T
                    prediction = 2 * torch.einsum("ni,ij,nj->n", delta, matrix, approximate_x) + torch.einsum("ni,ij,nj->n", delta, matrix, delta)
                    component_result[name] = metrics(actual.cpu().numpy(), prediction.cpu().numpy())
                results[corpus][stratum][str(component)] = component_result

    aggregate = {}
    for corpus, strata in results.items():
        aggregate[corpus] = {}
        for stratum, components in strata.items():
            aggregate[corpus][stratum] = {}
            for dimension in ("32", "64", "128"):
                aggregate[corpus][stratum][dimension] = {
                    key: float(np.median([row[dimension][key] for row in components.values()]))
                    for key in ("correlation", "centered_r2", "relative_error")
                }

    hypotheses = {
        "G1_high_score_generalization": {
            "supported": all(row["high"]["64"]["correlation"] >= 0.95 for row in aggregate.values()),
            "per_corpus": {corpus: row["high"]["64"]["correlation"] for corpus, row in aggregate.items()},
        },
        "G2_component_coverage": {
            "supported": all(
                sum(component["64"]["correlation"] >= 0.90 for component in results[corpus]["high"].values()) >= 14
                for corpus in results
            ),
            "per_corpus": {
                corpus: sum(component["64"]["correlation"] >= 0.90 for component in rows["high"].values())
                for corpus, rows in results.items()
            },
        },
        "G3_random_site_generalization": {
            "supported": all(row["random"]["64"]["correlation"] >= 0.90 for row in aggregate.values()),
            "per_corpus": {corpus: row["random"]["64"]["correlation"] for corpus, row in aggregate.items()},
        },
        "G4_context_gain": {
            "supported": all(
                row[stratum]["64"]["centered_r2"] - row[stratum]["32"]["centered_r2"] >= 0.20
                for row in aggregate.values() for stratum in ("high", "random")
            ),
            "gains": {
                corpus: {
                    stratum: row[stratum]["64"]["centered_r2"] - row[stratum]["32"]["centered_r2"]
                    for stratum in ("high", "random")
                }
                for corpus, row in aggregate.items()
            },
        },
        "G5_hybrid128_calibration": {
            "supported": all(row[stratum]["128"]["relative_error"] <= 0.10 for row in aggregate.values() for stratum in ("high", "random")),
            "values": {
                corpus: {stratum: row[stratum]["128"]["relative_error"] for stratum in ("high", "random")}
                for corpus, row in aggregate.items()
            },
        },
    }
    output = {"config": config, "aggregate": aggregate, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Generalization of the context-conditioned tangent interpreter",
        "",
        "The top-mode finite-effect calculation is extended to sixteen query-map components, high-score and random sites, and disjoint FineWeb/WikiText/TinyStories panels. Effects are algebraically exact full-form calculations; only the context x is compressed.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tangent generalization](query_map_tangent_generalization.png)",
        "",
        "| Corpus | Stratum | 32D corr / R² | 64D corr / R² | 128D corr / R² | 128D rel. error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, strata in aggregate.items():
        for stratum, dimensions in strata.items():
            lines.append(
                f"| {corpus} | {stratum} | {dimensions['32']['correlation']:.3f} / {dimensions['32']['centered_r2']:.3f} | "
                f"{dimensions['64']['correlation']:.3f} / {dimensions['64']['centered_r2']:.3f} | "
                f"{dimensions['128']['correlation']:.3f} / {dimensions['128']['centered_r2']:.3f} | "
                f"{dimensions['128']['relative_error']:.3f} |"
            )
    lines += [
        "",
        "Correlation measures directional prediction across sites; relative error also penalizes calibration. Random-site effects can be much smaller, so both statistics are required.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    dimensions = [32, 64, 128]
    for corpus, strata in aggregate.items():
        for stratum, style in (("high", "-"), ("random", "--")):
            plots[0].plot(dimensions, [strata[stratum][str(d)]["correlation"] for d in dimensions], linestyle=style, marker="o", label=f"{corpus} {stratum}")
            plots[1].plot(dimensions, [strata[stratum][str(d)]["relative_error"] for d in dimensions], linestyle=style, marker="o", label=f"{corpus} {stratum}")
    plots[0].set_ylabel("median component correlation")
    plots[0].set_ylim(-0.1, 1.02)
    plots[1].set_ylabel("median relative error")
    for plot in plots:
        plot.set_xlabel("context quotient dimension")
        plot.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
