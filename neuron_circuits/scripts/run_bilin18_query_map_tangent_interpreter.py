#!/usr/bin/env python3
"""Add context-dependent cross terms to the compact query-map interpreter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_input_basis_tradeoffs import ARRAYS as BASIS_ARRAYS
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_query_conditioned_interpreter import ARRAYS as MAP_ARRAYS, CONFIG as INTERPRETER_CONFIG, collect_states_and_writes
from run_bilin18_query_map_outlier_circuits import CONFIG as CIRCUIT_CONFIG, OUTPUT as CIRCUIT_OUTPUT
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_map_tangent_interpreter.json")
OUTPUT = TENSOR_ROOT / "query_map_tangent_interpreter.json"
REPORT = TENSOR_ROOT / "QUERY_MAP_TANGENT_INTERPRETER.md"
FIGURE = TENSOR_ROOT / "query_map_tangent_interpreter.png"


def stable_seed(seed: int, key: str) -> int:
    return seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    centered_actual = actual - actual.mean()
    centered_prediction = prediction - prediction.mean()
    correlation = float(np.corrcoef(actual, prediction)[0, 1])
    r2 = float(1 - np.square(centered_prediction - centered_actual).sum() / max(np.square(centered_actual).sum(), 1e-30))
    relative_error = float(np.linalg.norm(prediction - actual) / max(np.linalg.norm(actual), 1e-30))
    return {"correlation": correlation, "centered_r2": r2, "relative_error": relative_error}


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    circuit_config = json.loads(CIRCUIT_CONFIG.read_text())
    circuit_result = json.loads(CIRCUIT_OUTPUT.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[17]
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    saved_map = np.load(MAP_ARRAYS)
    output_components = torch.from_numpy(saved_map["output_components"][:, : circuit_config["components"]]).cuda()
    singular = torch.from_numpy(saved_map["singular_values"][: circuit_config["components"]]).cuda()
    restricted_matrices = singular[:, None, None] * torch.from_numpy(
        saved_map["matrix_components"][: circuit_config["components"]]
    ).cuda()

    # Exact full-space scalar forms for the four output readouts.
    full_matrices = []
    for component in range(circuit_config["components"]):
        coefficients = block.mlp.Down.weight.float().T @ output_components[:, component]
        matrix = block.mlp.Left.weight.float().T @ (
            coefficients[:, None] * block.mlp.Right.weight.float()
        )
        full_matrices.append(0.5 * (matrix + matrix.T))
    full_matrices = torch.stack(full_matrices)

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
    bases = {}
    for dimension in config["hybrid_dimensions"]:
        if dimension == 32:
            bases[str(dimension)] = semantic
        else:
            raw = torch.cat((semantic, state_v[:, : dimension - 32]), dim=1)
            bases[str(dimension)] = torch.linalg.qr(raw, mode="reduced").Q

    # Reconstruct the exact top-mode interventions used in the prior locked test.
    eigenvectors = []
    for component in range(circuit_config["components"]):
        values, vectors = torch.linalg.eigh(restricted_matrices[component])
        order = torch.argsort(values.abs(), descending=True)
        eigenvectors.append(axes @ vectors[:, order[: circuit_config["eigenspace_rank"]]])

    source = np.load(circuit_config["corpus"], mmap_mode="r")
    rows = source[circuit_config["start"] : circuit_config["start"] + circuit_config["rows"]]
    prior_top = [row for row in circuit_result["records"] if row["mode"] == "top"]
    records = []
    for prior in prior_top:
        component = prior["component"]
        row_index = prior["row"]
        position = prior["position"]
        raw = np.asarray(rows[row_index : row_index + 1, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, x_all = prefix_to_layer17(model, ids)
        x = x_all[0, position].float()
        vectors = eigenvectors[component]
        delta = -(x @ vectors) @ vectors.T
        delta *= circuit_config["edit_norm_fraction"] * x.norm() / delta.norm().clamp_min(1e-20)
        matrix = full_matrices[component]
        actual = float(2 * delta @ matrix @ x + delta @ matrix @ delta)
        predictions = {}
        for name, basis in bases.items():
            approximated_x = (x @ basis) @ basis.T
            predictions[name] = float(2 * delta @ matrix @ approximated_x + delta @ matrix @ delta)
        predictions["full"] = actual
        records.append({
            "component": component,
            "row": row_index,
            "position": position,
            "actual": actual,
            "prior_actual": prior["actual_component_change"],
            "predictions": predictions,
        })

    summaries = {}
    for component in range(circuit_config["components"]):
        selected = [row for row in records if row["component"] == component]
        actual = np.asarray([row["actual"] for row in selected])
        summaries[str(component)] = {}
        for name in list(bases) + ["full"]:
            prediction = np.asarray([row["predictions"][name] for row in selected])
            summaries[str(component)][name] = metrics(actual, prediction)
        prior_prediction = np.asarray([
            row["predictions"]["32"] for row in selected
        ])
        summaries[str(component)]["within32"] = metrics(actual, prior_prediction)

    prior_relative_error = max(
        abs(row["actual"] - row["prior_actual"]) for row in records
    ) / max(max(abs(row["prior_actual"]) for row in records), 1e-30)
    comp4 = summaries["3"]
    thresholds = {}
    for component, rows_for_component in summaries.items():
        thresholds[component] = next(
            (int(dimension) for dimension in config["hybrid_dimensions"] if rows_for_component[str(dimension)]["correlation"] >= 0.95),
            None,
        )
    hypotheses = {
        "T1_full_context_exact": {
            "supported": prior_relative_error < 1e-4,
            "maximum_relative_error": prior_relative_error,
        },
        "T2_hybrid64_repairs_component4": {
            "supported": comp4["64"]["correlation"] >= 0.80,
            "correlation": comp4["64"]["correlation"],
        },
        "T3_hybrid256_general_tangent": {
            "supported": all(row["256"]["correlation"] >= 0.90 for row in summaries.values()),
            "per_component": {component: row["256"]["correlation"] for component, row in summaries.items()},
        },
        "T4_outside_context_explains_failure": {
            "supported": comp4["full"]["correlation"] - comp4["32"]["correlation"] >= 0.50,
            "within32": comp4["32"]["correlation"],
            "full": comp4["full"]["correlation"],
            "gain": comp4["full"]["correlation"] - comp4["32"]["correlation"],
        },
        "T5_compact_tangent_threshold": {
            "supported": all(value is not None and value <= 512 for value in thresholds.values()),
            "minimum_dimension": thresholds,
        },
    }
    output = {"config": config, "summaries": summaries, "hypotheses": hypotheses, "records": records}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Context-conditioned tangent interpreter",
        "",
        "For an output query with symmetric bilinear form M and an edit δ inside the selected subspace, the exact scalar effect is `2 δᵀ M x + δᵀ M δ`. The earlier 32D interpreter replaces the full context x with its 32D projection and therefore omits cross terms with outside-state directions. This report restores increasingly large hybrid context quotients.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tangent interpreter](query_map_tangent_interpreter.png)",
        "",
        "| Component | 32D | 64D | 128D | 256D | 512D | Full |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for component, row in summaries.items():
        lines.append(
            f"| {int(component) + 1} | " + " | ".join(
                f"{row[name]['correlation']:.3f}" for name in ("32", "64", "128", "256", "512", "full")
            ) + " |"
        )
    lines += [
        "",
        "Cells are correlations across the same sixteen top-mode finite edits per component. The full formula is algebraically exact; compression quality measures how much context is needed to preserve the causal tangent, not the observational scalar score.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plot = plt.subplots(figsize=(7.5, 4.4))
    dimensions = config["hybrid_dimensions"]
    for component, row in summaries.items():
        plot.plot(dimensions, [row[str(dimension)]["correlation"] for dimension in dimensions], marker="o", label=f"component {int(component) + 1}")
    plot.axhline(0.95, color="black", linestyle=":", linewidth=1)
    plot.set_xlabel("context quotient dimension")
    plot.set_ylabel("finite-effect correlation")
    plot.set_ylim(-0.1, 1.02)
    plot.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
