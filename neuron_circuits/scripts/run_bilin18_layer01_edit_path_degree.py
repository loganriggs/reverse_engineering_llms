#!/usr/bin/env python3
"""Measure the polynomial degree of early-layer token-edit trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_prospective_tangent_router import (
    normalized_embedding,
    prefix_pre_from_x0,
    tangent_product,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_edit_path_degree.json")
OUTPUT = TENSOR_ROOT / "layer01_edit_path_degree.json"
REPORT = TENSOR_ROOT / "LAYER01_EDIT_PATH_DEGREE.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def design(nodes, degree, device, dtype):
    values = torch.tensor(nodes, device=device, dtype=dtype)
    return torch.stack([values.pow(power) for power in range(degree + 1)], dim=1)


def polynomial_metrics(train, evaluation, train_nodes, evaluation_nodes, degrees):
    """Fit vector paths on interleaved nodes and score delta-normalized error."""
    result = {}
    baseline = train[0]
    denominator = (evaluation - baseline).square().sum().clamp_min(1e-30)
    for degree in degrees:
        x_train = design(train_nodes, degree, train.device, train.dtype)
        x_eval = design(evaluation_nodes, degree, train.device, train.dtype)
        coefficients = torch.linalg.pinv(x_train) @ train
        prediction = x_eval @ coefficients
        fidelity = 1 - (prediction - evaluation).square().sum() / denominator
        result[str(degree)] = float(fidelity)
        if degree == 2:
            result["quadratic_share"] = float(
                coefficients[2].norm() /
                (coefficients[1].norm() + coefficients[2].norm()).clamp_min(1e-30)
            )
    all_values = torch.cat([train, evaluation], dim=0)
    centered = all_values - all_values.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    result["rank2_energy"] = float(singular[:2].square().sum() / singular.square().sum().clamp_min(1e-30))
    return result


def path_values(model, block, ids, target_token, layer, edit_position, final_position, nodes):
    x0 = normalized_embedding(model, ids)
    target_id = torch.tensor([target_token], device=ids.device)
    target = normalized_embedding(model, target_id)[0]
    direction = torch.zeros_like(x0)
    direction[:, edit_position] = target - x0[:, edit_position]
    outputs = {"pre": [], "product": [], "write": []}
    for node in nodes:
        pre = prefix_pre_from_x0(model, x0 + node * direction, layer, final_position)[0]
        state = F.rms_norm(pre, (model.config.n_embd,))
        product = block.mlp.Left(state) * block.mlp.Right(state)
        write = F.linear(product, block.mlp.Down.weight.float())
        outputs["pre"].append(pre)
        outputs["product"].append(product)
        outputs["write"].append(write)
    return {name: torch.stack(values) for name, values in outputs.items()}


@torch.no_grad()
def main():
    config = json.loads(CONFIG.read_text())
    redteam = json.loads(Path(config["source_config"]).read_text())
    source = json.loads(Path(redteam["source_config"]).read_text())
    data_config = json.loads(Path(source["data_config"]).read_text())
    pairs = json.loads(Path(source["evaluation_pair_config"]).read_text())
    edit_position, final_position = data_config["edit_position"], data_config["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in data_config["evaluation"].items()
    }
    all_nodes = config["train_nodes"] + config["evaluation_nodes"]
    records = {str(layer): [] for layer in source["layers"]}
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        generator = np.random.default_rng(config["seed"] + corpus_index)
        for family, lexical_pairs in pairs["new_families"].items():
            for source_token, target_token in lexical_pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    config["contexts_per_pair"], generator,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(
                        rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    ids = torch.as_tensor(raw, device="cuda")[None]
                    for layer in source["layers"]:
                        block = model.blocks[layer]
                        values = path_values(
                            model, block, ids, target_token, layer, edit_position,
                            final_position, all_nodes,
                        )
                        count = len(config["train_nodes"])
                        metrics = {
                            name: polynomial_metrics(
                                value[:count], value[count:], config["train_nodes"],
                                config["evaluation_nodes"], config["degrees"],
                            )
                            for name, value in values.items()
                        }
                        midpoint, _ = tangent_product(
                            model, block, ids, target_token, layer, edit_position,
                            final_position, path_point=0.5,
                        )
                        endpoint = values["product"][config["train_nodes"].index(1.0)] - values["product"][0]
                        midpoint_error = float(
                            (midpoint - endpoint).norm() / endpoint.norm().clamp_min(1e-30)
                        )
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "metrics": metrics,
                            "midpoint_endpoint_error": midpoint_error,
                        })
                print("edit path degree", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    for layer in source["layers"]:
        summaries[str(layer)] = {}
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {
                "midpoint_endpoint_error": median([row["midpoint_endpoint_error"] for row in selected]),
                "metrics": {
                    kind: {
                        key: median([row["metrics"][kind][key] for row in selected])
                        for key in [str(degree) for degree in config["degrees"]] + ["quadratic_share", "rank2_energy"]
                    }
                    for kind in ("pre", "product", "write")
                },
            }

    def row(layer, corpus):
        return summaries[str(layer)][corpus]

    hypotheses = {
        "P0_product_path_is_quadratic": {
            "supported": all(row(layer, corpus)["metrics"]["product"]["2"] >= 0.95 for layer in source["layers"] for corpus in arrays)
        },
        "P1_quadratic_materially_beats_linear": {
            "supported": all(row(layer, corpus)["metrics"]["product"]["2"] - row(layer, corpus)["metrics"]["product"]["1"] >= 0.20 for layer in source["layers"] for corpus in arrays)
        },
        "P2_cubic_adds_little": {
            "supported": all(row(layer, corpus)["metrics"]["product"]["3"] - row(layer, corpus)["metrics"]["product"]["2"] <= 0.03 for layer in source["layers"] for corpus in arrays)
        },
        "P3_product_path_is_nearly_planar": {
            "supported": all(row(layer, corpus)["metrics"]["product"]["rank2_energy"] >= 0.99 for layer in source["layers"] for corpus in arrays)
        },
        "P4_midpoint_derivative_is_a_secant_program": {
            "supported": all(row(layer, corpus)["midpoint_endpoint_error"] <= 0.20 for layer in source["layers"] for corpus in arrays)
        },
        "P5_quadratic_write_description_transfers_ood": {
            "supported": all(abs(row(layer, "fineweb")["metrics"]["write"]["2"] - row(layer, "tinystories")["metrics"]["write"]["2"]) <= 0.05 for layer in source["layers"])
        },
        "P6_curvature_is_material": {
            "supported": all(row(layer, corpus)["metrics"]["product"]["quadratic_share"] >= 0.15 for layer in source["layers"] for corpus in arrays)
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": records,
    }, indent=2))
    lines = [
        "# Polynomial degree of Layer-0/1 token-edit trajectories",
        "",
        "Vector-valued paths are sampled at five fit nodes and four held-out interleaved nodes along the straight source-to-target embedding interpolation. Per-example polynomial coefficients are fit jointly across all vector coordinates. Fidelity is normalized by held-out delta energy from the unedited state, so the large unchanged baseline cannot make the score look artificially good.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Layer | Corpus | Product polynomial d1/d2/d3/d4 | Pre d2 | Write d2 | Product rank-2 energy | Quadratic coefficient share | Midpoint secant error |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in source["layers"]:
        for corpus in arrays:
            item = row(layer, corpus)
            product = item["metrics"]["product"]
            lines.append(
                f"| {layer} | {corpus} | {product['1']:.3f}/{product['2']:.3f}/{product['3']:.3f}/{product['4']:.3f} | "
                f"{item['metrics']['pre']['2']:.3f} | {item['metrics']['write']['2']:.3f} | "
                f"{product['rank2_energy']:.4f} | {product['quadratic_share']:.3f} | "
                f"{item['midpoint_endpoint_error']:.3f} |"
            )
    lines += [
        "",
        "A low-degree path is an edit-local description, not a global semantic coordinate system. Its use is to compile a specified finite edit into native execution coordinates; it does not imply that unrelated edits share neuron identities.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
