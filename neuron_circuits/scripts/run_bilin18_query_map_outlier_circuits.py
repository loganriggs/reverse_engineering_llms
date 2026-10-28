#!/usr/bin/env python3
"""Turn the leading query-map spectral components into finite circuit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_query_conditioned_interpreter import ARRAYS as MAP_ARRAYS
from run_bilin18_query_map_spectral_redteam import OUTPUT as SPECTRAL_OUTPUT
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_map_outlier_circuits.json")
OUTPUT = TENSOR_ROOT / "query_map_outlier_circuits.json"
REPORT = TENSOR_ROOT / "QUERY_MAP_OUTLIER_CIRCUITS.md"
FIGURE = TENSOR_ROOT / "query_map_outlier_circuits.png"


def stable_seed(seed: int, key: str) -> int:
    return seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def correlation(first: list[float], second: list[float]) -> float:
    return float(np.corrcoef(first, second)[0, 1])


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    spectral = json.loads(SPECTRAL_OUTPUT.read_text())
    saved = np.load(MAP_ARRAYS)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    dimension = model.config.n_embd
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    output_components = torch.from_numpy(saved["output_components"][:, : config["components"]]).cuda()
    singular = torch.from_numpy(saved["singular_values"][: config["components"]]).cuda()
    matrices = torch.from_numpy(saved["matrix_components"][: config["components"]]).cuda()
    matrices = singular[:, None, None] * matrices

    component_data = []
    for index in range(config["components"]):
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices[index])
        order = torch.argsort(eigenvalues.abs(), descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        energy = eigenvalues.square()
        top_energy = float(energy[: config["eigenspace_rank"]].sum() / energy.sum())
        top_input = axes @ eigenvectors[:, : config["eigenspace_rank"]]
        low_input = axes @ eigenvectors[:, -config["eigenspace_rank"] :]
        generator = torch.Generator(device="cpu").manual_seed(stable_seed(config["seed"], f"component-{index}"))
        random_coordinates = torch.randn(32, config["eigenspace_rank"], generator=generator)
        random_coordinates = torch.linalg.qr(random_coordinates, mode="reduced").Q.cuda()
        random_input = axes @ random_coordinates
        component_data.append({
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "top_input": top_input,
            "low_input": low_input,
            "random_input": random_input,
            "top4_energy": top_energy,
        })

    source = np.load(config["corpus"], mmap_mode="r")
    rows = source[config["start"] : config["start"] + config["rows"]]
    total = len(rows) * len(config["positions"])
    full_scores = np.empty((total, config["components"]), dtype=np.float32)
    predicted_scores = np.empty_like(full_scores)
    site_rows = np.empty(total, dtype=np.int64)
    site_positions = np.empty(total, dtype=np.int64)
    offset = 0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, x = prefix_to_layer17(model, ids)
        selected = x[:, config["positions"]].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(selected) * block.mlp.Right(selected))
        z = selected @ axes
        count = len(selected)
        full_scores[offset : offset + count] = (write @ output_components).cpu().numpy()
        predicted_scores[offset : offset + count] = torch.einsum("ni,kij,nj->nk", z, matrices, z).cpu().numpy()
        batch_rows = np.repeat(np.arange(start, start + len(raw)), len(config["positions"]))
        site_rows[offset : offset + count] = batch_rows
        site_positions[offset : offset + count] = np.tile(config["positions"], len(raw))
        offset += count

    selected_sites = {}
    examples = {}
    for component in range(config["components"]):
        centered = full_scores[:, component] - np.median(full_scores[:, component])
        indices = np.argpartition(np.abs(centered), -config["sites_per_component"])[-config["sites_per_component"] :]
        indices = indices[np.argsort(np.abs(centered[indices]))[::-1]]
        selected_sites[component] = indices.tolist()
        positive = np.argsort(full_scores[:, component])[-6:][::-1]
        negative = np.argsort(full_scores[:, component])[:6]
        examples[component] = {}
        for tail, selected in (("positive", positive), ("negative", negative)):
            examples[component][tail] = []
            for site_index in selected:
                row_index = int(site_rows[site_index])
                position = int(site_positions[site_index])
                context_ids = np.asarray(rows[row_index, max(0, position - 12) : min(128, position + 1)], dtype=np.int64)
                examples[component][tail].append({
                    "score": float(full_scores[site_index, component]),
                    "token": tokenizer.decode([int(rows[row_index, position])]),
                    "context": tokenizer.decode(context_ids.tolist()),
                })

    records = []
    for component, indices in selected_sites.items():
        data = component_data[component]
        output_direction = output_components[:, component]
        matrix = matrices[component]
        for site_index in indices:
            row_index = int(site_rows[site_index])
            position = int(site_positions[site_index])
            raw = np.asarray(rows[row_index : row_index + 1, :128], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")
            pre, x_all = prefix_to_layer17(model, ids)
            x = x_all[0, position].float()
            pre_site = pre[0, position].float()
            native_product = block.mlp.Left(x) * block.mlp.Right(x)
            native_write = block.mlp.Down(native_product) + block.mlp.Down_bias
            native_residual = pre_site + native_write
            native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(native_residual, (dimension,))).float() / 30)
            native_log_prob = native_logits.log_softmax(-1)
            native_probability = native_log_prob.exp()
            native_component_score = float(native_write @ output_direction)
            z = x @ axes
            native_predicted = float(z @ matrix @ z)
            for mode, vectors in (
                ("top", data["top_input"]),
                ("low", data["low_input"]),
                ("random", data["random_input"]),
            ):
                delta = -(x @ vectors) @ vectors.T
                delta *= config["edit_norm_fraction"] * x.norm() / delta.norm().clamp_min(1e-20)
                changed_x = x + delta
                changed_write = block.mlp.Down(block.mlp.Left(changed_x) * block.mlp.Right(changed_x)) + block.mlp.Down_bias
                changed_residual = pre_site + changed_write
                changed_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(changed_residual, (dimension,))).float() / 30)
                changed_log_prob = changed_logits.log_softmax(-1)
                changed_z = changed_x @ axes
                records.append({
                    "component": component,
                    "row": row_index,
                    "position": position,
                    "mode": mode,
                    "actual_component_change": float(changed_write @ output_direction) - native_component_score,
                    "predicted_component_change": float(changed_z @ matrix @ changed_z) - native_predicted,
                    "kl": float((native_probability * (native_log_prob - changed_log_prob)).sum()),
                    "delta_norm": float(delta.norm()),
                })

    summaries = {}
    for component in range(config["components"]):
        summaries[str(component)] = {
            "top4_matrix_energy": component_data[component]["top4_energy"],
            "score_correlation_on_scan": correlation(
                full_scores[:, component].tolist(), predicted_scores[:, component].tolist()
            ),
            "output_profile": spectral["semantic_profiles"][component],
            "examples": examples[component],
            "modes": {},
        }
        for mode in ("top", "low", "random"):
            selected = [row for row in records if row["component"] == component and row["mode"] == mode]
            actual = [row["actual_component_change"] for row in selected]
            predicted = [row["predicted_component_change"] for row in selected]
            summaries[str(component)]["modes"][mode] = {
                "median_absolute_component_change": float(np.median(np.abs(actual))),
                "median_kl": float(np.median([row["kl"] for row in selected])),
                "prediction_correlation": correlation(actual, predicted),
            }

    hypotheses = {
        "O1_low_rank_input_forms": {
            "supported": all(row["top4_matrix_energy"] >= 0.50 for row in summaries.values()),
            "per_component": {key: row["top4_matrix_energy"] for key, row in summaries.items()},
        },
        "O2_top_modes_specific": {
            "supported": all(
                row["modes"]["top"]["median_absolute_component_change"]
                >= 2 * row["modes"]["low"]["median_absolute_component_change"]
                for row in summaries.values()
            ),
            "ratios": {
                key: row["modes"]["top"]["median_absolute_component_change"]
                / max(row["modes"]["low"]["median_absolute_component_change"], 1e-30)
                for key, row in summaries.items()
            },
        },
        "O3_weight_prediction_matches_finite_effect": {
            "supported": all(row["modes"]["top"]["prediction_correlation"] >= 0.80 for row in summaries.values()),
            "per_component": {key: row["modes"]["top"]["prediction_correlation"] for key, row in summaries.items()},
        },
        "O4_top_modes_reach_logits": {
            "supported": all(
                row["modes"]["top"]["median_kl"] >= 2 * row["modes"]["low"]["median_kl"]
                for row in summaries.values()
            ),
            "ratios": {
                key: row["modes"]["top"]["median_kl"] / max(row["modes"]["low"]["median_kl"], 1e-30)
                for key, row in summaries.items()
            },
        },
        "O5_random_control": {
            "supported": sum(
                row["modes"]["top"]["median_absolute_component_change"]
                > row["modes"]["random"]["median_absolute_component_change"]
                for row in summaries.values()
            ) >= 3,
            "components_top_larger": sum(
                row["modes"]["top"]["median_absolute_component_change"]
                > row["modes"]["random"]["median_absolute_component_change"]
                for row in summaries.values()
            ),
        },
    }
    output = {"config": config, "summaries": summaries, "hypotheses": hypotheses, "records": records}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Finite circuit tests for the leading query-map components",
        "",
        "The leading four readout→interaction singular components are expanded into output token roles, 32D input interaction matrices, natural context tails, and norm-matched finite edits of top, low, and random four-dimensional input eigenspaces.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Outlier circuits](query_map_outlier_circuits.png)",
        "",
        "| Component | Best output predicate | Output AUROC | Top-4 matrix energy | Scan corr | Top/low scalar effect | Top/low KL | Finite prediction corr |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summaries.items():
        top, low = row["modes"]["top"], row["modes"]["low"]
        lines.append(
            f"| {int(key) + 1} | {row['output_profile']['best_concept']} | {row['output_profile']['best_oriented_auc']:.3f} | "
            f"{row['top4_matrix_energy']:.3f} | {row['score_correlation_on_scan']:.3f} | "
            f"{top['median_absolute_component_change'] / max(low['median_absolute_component_change'], 1e-30):.2f}× | "
            f"{top['median_kl'] / max(low['median_kl'], 1e-30):.2f}× | {top['prediction_correlation']:.3f} |"
        )
    lines += [
        "",
        "Full context tails and promoted/suppressed token examples are stored in `query_map_outlier_circuits.json`. These are candidate circuits only at the declared output-component readout; null comparisons showed that broad token-predicate AUROC alone is not sufficient evidence.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2))
    components = np.arange(1, config["components"] + 1)
    width = 0.25
    for offset, mode in ((-width, "top"), (0, "low"), (width, "random")):
        plots[0].bar(
            components + offset,
            [summaries[str(index)]["modes"][mode]["median_absolute_component_change"] for index in range(config["components"])],
            width,
            label=mode,
        )
        plots[1].bar(
            components + offset,
            [summaries[str(index)]["modes"][mode]["median_kl"] for index in range(config["components"])],
            width,
            label=mode,
        )
    plots[0].set_ylabel("median absolute component-score change")
    plots[1].set_ylabel("median final-logit KL")
    for plot in plots:
        plot.set_xlabel("query-map component")
        plot.set_xticks(components)
        plot.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
