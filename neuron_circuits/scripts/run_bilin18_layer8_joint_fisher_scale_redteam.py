#!/usr/bin/env python3
"""Finite-scale red-team for the joint downstream-Fisher program."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    prefix_to_layer,
    repeat_batch,
    suffix_from_layer,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_joint_fisher_scale_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_joint_fisher_scale_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_JOINT_FISHER_SCALE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_joint_fisher_scale_redteam.png"


@torch.inference_mode()
def run_routes(model, pre, normalized, x0, value_bus, position, mode_directions,
               indices, output_direction, scale):
    names = list(indices)
    block = model.blocks[8]
    x = normalized[0, position].float()
    variants = normalized.repeat(len(names) + 1, 1, 1)
    for variant, name in enumerate(names, start=1):
        directions = mode_directions[:, indices[name]]
        raw_delta = -(x @ directions) @ directions.T
        delta = raw_delta * (scale * x.norm() / raw_delta.norm().clamp_min(1e-20))
        variants[variant, position] = (x + delta).to(variants.dtype)
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(variants)) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(variants)), repeat_batch(value_bus, len(variants)),
        outputs, position,
    )
    native = logits[0]
    native_write = writes[0, position].float()
    result = {}
    for variant, name in enumerate(names, start=1):
        result[name] = {
            "native_kl": exact_kl(native, logits[variant]),
            "absolute_direct_query_effect": abs(float(
                (writes[variant, position].float() - native_write) @ output_direction
            )),
            "effect": logits[variant] - native,
        }
    return native, result


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    discovery_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, discovery_write = collect(model, 8, source_config["discovery"], discovery_config)
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, :source_config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :source_config["queries"]]
    block = model.blocks[8]
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    _, vectors = ordered_eigensystem(matrices)

    results = {}
    route_names = source_config["routes"]
    for corpus, source_result in source["results"].items():
        corpus_config = source_config["evaluation"][corpus]
        corpus_source = np.load(corpus_config["path"], mmap_mode="r")
        rows = corpus_source[
            corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]
        ]
        records = []
        for record in source_result["records"]:
            query = record["query"]
            ids = torch.as_tensor(
                np.asarray(rows[record["row"], :source_config["causal_position"] + 1], dtype=np.int64),
                device="cuda",
            )[None]
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
            mode_directions = basis @ vectors[query]
            indices = {
                name: torch.as_tensor(record["routes"][name]["indices"], device="cuda")
                for name in route_names
            }
            scales = {}
            reference = None
            for scale in config["edit_norm_fractions"]:
                native, route_rows = run_routes(
                    model, pre, normalized, x0, value_bus, source_config["causal_position"],
                    mode_directions, indices, directions[:, query], scale,
                )
                if reference is None:
                    reference = {name: row["effect"] for name, row in route_rows.items()}
                for name, row in route_rows.items():
                    _, cosine, _ = fisher_fidelity(native, reference[name], row["effect"])
                    row["cosine_to_one_percent"] = cosine
                    row.pop("effect")
                scales[str(scale)] = route_rows
            records.append({"query": query, "row": record["row"], "scales": scales})
        summary = {}
        for scale in config["edit_norm_fractions"]:
            scale_rows = {}
            for name in route_names:
                scale_rows[name] = {
                    key: float(np.median([
                        record["scales"][str(scale)][name][key] for record in records
                    ]))
                    for key in ("native_kl", "absolute_direct_query_effect", "cosine_to_one_percent")
                }
            summary[str(scale)] = scale_rows
        results[corpus] = {"summary": summary, "records": records}
        print("completed", corpus, flush=True)

    def row(corpus, scale, route):
        return results[corpus]["summary"][str(scale)][route]

    corpora = list(results)
    scales = config["edit_norm_fractions"]
    hypotheses = {
        "S1_joint_advantage_is_not_a_single_scale_artifact": {
            "supported": all(
                row(c, s, "fisher_joint_greedy")["native_kl"]
                >= 3 * row(c, s, "local_oracle")["native_kl"]
                for c in corpora for s in scales
            ),
        },
        "S2_joint_effect_direction_is_scale_stable": {
            "supported": all(
                row(c, s, "fisher_joint_greedy")["cosine_to_one_percent"]
                >= (0.75 if s == 0.2 else 0.90)
                for c in corpora for s in scales
            ),
        },
        "S3_joint_program_beats_unit_diagonal": {
            "supported": all(
                row(c, s, "fisher_joint_greedy")["native_kl"]
                >= 2 * row(c, s, "fisher_unit")["native_kl"]
                for c in corpora for s in scales
            ),
        },
        "S4_joint_effect_is_monotonic": {
            "supported": all(
                all(
                    row(c, scales[i + 1], "fisher_joint_greedy")["native_kl"]
                    > row(c, scales[i], "fisher_joint_greedy")["native_kl"]
                    for i in range(len(scales) - 1)
                ) for c in corpora
            ),
        },
        "S5_local_query_tradeoff_persists": {
            "supported": all(
                row(c, s, "local_oracle")["absolute_direct_query_effect"]
                > row(c, s, "fisher_joint_greedy")["absolute_direct_query_effect"]
                for c in corpora for s in scales
            ),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Scale red-team for the joint Fisher program",
        "",
        "The frozen per-example mode sets from the 5%-norm joint-Fisher experiment are rerun at 1%, 5%, 10%, and 20% input norm. No mode selection or Fisher estimation is repeated at the new scales.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Joint Fisher scale red-team](layer8_joint_fisher_scale_redteam.png)",
        "",
        "| Corpus | Scale | Local KL | Diagonal KL | Unit-diagonal KL | Joint KL | Joint/local | Joint effect cosine to 1% | Local/joint query effect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        for scale in scales:
            local = row(corpus, scale, "local_oracle")
            diagonal = row(corpus, scale, "fisher_diagonal")
            unit = row(corpus, scale, "fisher_unit")
            joint = row(corpus, scale, "fisher_joint_greedy")
            lines.append(
                f"| {corpus} | {scale:.2f} | {local['native_kl']:.3e} | "
                f"{diagonal['native_kl']:.3e} | {unit['native_kl']:.3e} | {joint['native_kl']:.3e} | "
                f"{joint['native_kl'] / max(local['native_kl'], 1e-30):.2f}× | "
                f"{joint['cosine_to_one_percent']:.3f} | "
                f"{local['absolute_direct_query_effect'] / max(joint['absolute_direct_query_effect'], 1e-30):.2f}× |"
            )
    lines += [
        "",
        "This checks finite-scale robustness at the same states. It does not turn the per-example oracle into a reusable router or prove semantic coherence.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus in corpora:
        axes[0].plot(scales, [row(corpus, s, "fisher_joint_greedy")["native_kl"] for s in scales], marker="o", label=f"{corpus}/joint")
        axes[0].plot(scales, [row(corpus, s, "local_oracle")["native_kl"] for s in scales], marker="x", linestyle="--", label=f"{corpus}/local")
        axes[1].plot(scales, [row(corpus, s, "fisher_joint_greedy")["cosine_to_one_percent"] for s in scales], marker="o", label=corpus)
    axes[0].set_title("Exact finite effect")
    axes[0].set_ylabel("median native KL")
    axes[1].set_title("Joint effect direction")
    axes[1].set_ylabel("Fisher cosine to 1% effect")
    for axis in axes:
        axis.set_xlabel("input edit norm fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
