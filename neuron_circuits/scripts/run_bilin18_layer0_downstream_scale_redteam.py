#!/usr/bin/env python3
"""Scale red-team of layer-0 joint-axis downstream role complexity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_layer0_downstream_roles import (
    OUTPUT as SOURCE,
    directional_summary,
    layer0_prefix,
    propagate,
    repeat_batch,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer0_downstream_scale_redteam.json")
OUTPUT = TENSOR_ROOT / "layer0_downstream_scale_redteam.json"
REPORT = TENSOR_ROOT / "LAYER0_DOWNSTREAM_SCALE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer0_downstream_scale_redteam.png"


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads(SOURCE.read_text())
    source_config = source["config"]
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer0_axes"]).cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block0 = model.blocks[0]
    corpus = np.load(source_config["corpus"], mmap_mode="r")
    rows = corpus[
        source_config["start"] : source_config["start"] + source_config["scan_rows"]
    ]
    selected_axes = [row["axis"] for row in source["axes"]]
    requests_by_row: dict[int, list[int]] = {}
    for record in source["records"]:
        requests_by_row.setdefault(record["row"], []).append(record["axis"])
    requests_by_row = {
        row: sorted(set(requests)) for row, requests in requests_by_row.items()
    }

    local_effects = {
        scale: {axis: [] for axis in selected_axes} for scale in config["scales"]
    }
    final_effects = {
        scale: {axis: [] for axis in selected_axes} for scale in config["scales"]
    }
    effects_by_site = {}
    records = []
    position = source_config["position"]
    for progress, (row_index, requested_axes) in enumerate(sorted(requests_by_row.items())):
        raw = np.asarray(rows[row_index : row_index + 1, : source_config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, pre, mlp_input, _, write, native_x = layer0_prefix(model, ids)
        variants = [native_x]
        metadata = []
        x = mlp_input[0, position]
        for scale in config["scales"]:
            for axis in requested_axes:
                direction = axes[:, axis]
                changed_input = x - scale * (x @ direction) * direction
                changed_product = block0.mlp.Left(changed_input) * block0.mlp.Right(changed_input)
                changed_write = block0.mlp.Down(changed_product) + block0.mlp.Down_bias
                changed_x = native_x.clone()
                changed_x[0, position] = pre[0, position] + changed_write
                variants.append(changed_x)
                metadata.append((scale, axis, changed_write - write[0, position]))
        batch_count = len(variants)
        final = propagate(
            model,
            torch.cat(variants, dim=0),
            repeat_batch(value_bus, batch_count),
            repeat_batch(x0, batch_count),
        )[:, position]
        logits = 30 * torch.tanh(model.lm_head(final).float() / 30)
        native_log_prob = logits[0].log_softmax(-1)
        native_probability = native_log_prob.exp()
        for variant_index, (scale, axis, local_delta) in enumerate(metadata, start=1):
            final_delta = final[variant_index] - final[0]
            changed_log_prob = logits[variant_index].log_softmax(-1)
            kl = float((native_probability * (native_log_prob - changed_log_prob)).sum())
            local_array = local_delta.cpu().numpy()
            final_array = final_delta.cpu().numpy()
            local_effects[scale][axis].append(local_array)
            final_effects[scale][axis].append(final_array)
            effects_by_site[(row_index, axis, scale)] = final_array
            records.append({"row": row_index, "axis": axis, "scale": scale, "kl": kl})
        if progress % 50 == 0:
            print(progress + 1, "/", len(requests_by_row), flush=True)

    summaries = {}
    for scale in config["scales"]:
        summaries[str(scale)] = {}
        for stage, values in (("local", local_effects), ("final", final_effects)):
            axis_rows = {
                str(axis): directional_summary(np.asarray(values[scale][axis]))
                for axis in selected_axes
            }
            summaries[str(scale)][stage] = {
                "first_component_energy": float(np.median([row["first_component_energy"] for row in axis_rows.values()])),
                "four_component_energy": float(np.median([row["four_component_energy"] for row in axis_rows.values()])),
                "rank90": float(np.median([row["rank90"] for row in axis_rows.values()])),
                "axes": axis_rows,
            }
        summaries[str(scale)]["median_kl"] = float(np.median([
            row["kl"] for row in records if row["scale"] == scale
        ]))

    smallest, largest = config["scales"][0], config["scales"][-1]
    direction_cosines = [
        cosine(
            effects_by_site[(row_index, axis, smallest)],
            effects_by_site[(row_index, axis, largest)],
        )
        for row_index, axes_for_row in requests_by_row.items()
        for axis in axes_for_row
    ]
    scale_values = np.asarray(config["scales"], dtype=float)
    kl_values = np.asarray([summaries[str(scale)]["median_kl"] for scale in config["scales"]])
    kl_slope = float(np.polyfit(np.log(scale_values), np.log(np.maximum(kl_values, 1e-30)), 1)[0])
    small_summary = summaries[str(smallest)]
    rank_change = abs(
        summaries[str(largest)]["final"]["rank90"] - small_summary["final"]["rank90"]
    ) / small_summary["final"]["rank90"]
    hypotheses = {
        "S1_tangent_rank_survives": {
            "supported": small_summary["final"]["rank90"] >= small_summary["local"]["rank90"],
            "local": small_summary["local"]["rank90"],
            "final": small_summary["final"]["rank90"],
        },
        "S2_final_rank_scale_stability": {
            "supported": rank_change <= 0.20,
            "relative_change": rank_change,
        },
        "S3_direction_scale_stability": {
            "supported": float(np.median(direction_cosines)) >= 0.70,
            "median_cosine": float(np.median(direction_cosines)),
        },
        "S4_local_kl_scaling": {
            "supported": 1.5 <= kl_slope <= 2.5,
            "slope": kl_slope,
            "median_kl": {str(scale): summaries[str(scale)]["median_kl"] for scale in config["scales"]},
        },
    }
    output = {
        "config": config,
        "summaries": summaries,
        "direction_cosines": {
            "median": float(np.median(direction_cosines)),
            "q10": float(np.quantile(direction_cosines, 0.1)),
            "q90": float(np.quantile(direction_cosines, 0.9)),
        },
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Scale red-team of layer-0 downstream role complexity",
        "",
        "The identical 384 selected axis/site pairs are rerun at 10%, 25%, and 100% coordinate removal. This distinguishes context-dependent downstream Jacobians from complexity created only by large nonlinear edits.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Scale red-team](layer0_downstream_scale_redteam.png)",
        "",
        "| Scale | Local first / four | Local rank90 | Final first / four | Final rank90 | Median KL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for scale in config["scales"]:
        row = summaries[str(scale)]
        lines.append(
            f"| {scale:.2f} | {row['local']['first_component_energy']:.3f} / {row['local']['four_component_energy']:.3f} | "
            f"{row['local']['rank90']:.1f} | {row['final']['first_component_energy']:.3f} / "
            f"{row['final']['four_component_energy']:.3f} | {row['final']['rank90']:.1f} | {row['median_kl']:.3g} |"
        )
    lines += [
        "",
        f"Median final-effect direction cosine between 10% and 100% removal is {np.median(direction_cosines):.3f}; the median-KL scale exponent is {kl_slope:.2f}.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2))
    plots[0].plot(config["scales"], [summaries[str(s)]["local"]["rank90"] for s in config["scales"]], marker="o", label="local")
    plots[0].plot(config["scales"], [summaries[str(s)]["final"]["rank90"] for s in config["scales"]], marker="o", label="final")
    plots[0].set_xlabel("coordinate-removal scale")
    plots[0].set_ylabel("median rank90")
    plots[0].legend()
    plots[1].loglog(config["scales"], kl_values, marker="o")
    plots[1].set_xlabel("coordinate-removal scale")
    plots[1].set_ylabel("median next-token KL")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
