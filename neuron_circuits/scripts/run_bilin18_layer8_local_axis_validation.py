#!/usr/bin/env python3
"""Finite causal validation of decoded basis-invariant Layer-8 axes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_direct_tensor_jacobian import (
    analytic_suffix,
    bilinear_mlp_jvp,
    output_fisher_response,
)
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    prefix_to_layer,
    repeat_batch,
    suffix_from_layer,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_local_axis_validation.json")
OUTPUT = TENSOR_ROOT / "layer8_local_axis_validation.json"
REPORT = TENSOR_ROOT / "LAYER8_LOCAL_AXIS_VALIDATION.md"
FIGURE = TENSOR_ROOT / "layer8_local_axis_validation.png"


def median(rows, key):
    return float(np.median([row[key] for row in rows]))


@torch.inference_mode()
def finite_directions(model, block, pre, normalized, x0, value_bus, position,
                      directions, scales):
    native_product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
    native_write = block.mlp.Down(native_product) + block.mlp.Down_bias
    native_state = pre + native_write
    native_logits = suffix_from_layer(model, 8, x0, value_bus, native_state, position)[0]
    x = normalized[0, position].float()
    results = {}
    for scale in scales:
        count = directions.shape[1]
        variants = normalized.repeat(count, 1, 1)
        variants[:, position] = (x[None] + scale * x.norm() * directions.T).to(variants.dtype)
        writes = block.mlp.Down(block.mlp.Left(variants) * block.mlp.Right(variants)) + block.mlp.Down_bias
        states = repeat_batch(pre, count) + writes
        logits = suffix_from_layer(
            model, 8, repeat_batch(x0, count),
            repeat_batch(value_bus, count), states, position,
        )
        results[scale] = (native_logits, logits)
    return results


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, _ = collect(model, 8, source_config["tensor_discovery"], tensor_config)
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = tensor_vh.T[:, :source_config["dimension"]]
    block = model.blocks[8]
    position = source_config["causal_position"]
    results = {}

    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        sites = rng.choice(len(rows), config["sites_per_corpus"], replace=False)
        records = []
        for site_number, row_index in enumerate(sites):
            ids = torch.as_tensor(
                np.asarray(rows[row_index, :position + 1], dtype=np.int64), device="cuda",
            )[None]
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
            native_write = block.mlp.Down(
                block.mlp.Left(normalized) * block.mlp.Right(normalized)
            ) + block.mlp.Down_bias
            x8 = pre + native_write
            x = normalized[0, position].float()
            unit_write_tangents = bilinear_mlp_jvp(block, x, basis.T)
            logits, logit_tangents, _ = analytic_suffix(
                model, x8, x0, value_bus, position, unit_write_tangents,
                include_attention=True,
            )
            response = output_fisher_response(logits, logit_tangents)
            _, gram_vectors = gram_eigensystem(response)
            top_coefficients = gram_vectors[:, :config["axes"]]
            generator = torch.Generator(device="cuda").manual_seed(
                config["seed"] + corpus_index * 10000 + site_number,
            )
            random_coefficients = torch.linalg.qr(torch.randn(
                source_config["dimension"], config["random_controls"],
                generator=generator, device="cuda",
            ), mode="reduced").Q
            coefficients = torch.cat((top_coefficients, random_coefficients), dim=1)
            directions = basis @ coefficients
            finite = finite_directions(
                model, block, pre, normalized, x0, value_bus, position,
                directions, config["edit_norm_fractions"],
            )
            site_rows = []
            for scale in config["edit_norm_fractions"]:
                native_logits, changed_logits = finite[scale]
                native_probability = F.softmax(native_logits.float(), -1)
                tangent_effects = scale * x.norm() * (coefficients.T @ logit_tangents)
                scale_rows = []
                for direction_index in range(coefficients.shape[1]):
                    actual_effect = changed_logits[direction_index] - native_logits
                    fidelity, cosine, _ = fisher_fidelity(
                        native_logits, actual_effect, tangent_effects[direction_index],
                    )
                    row = {
                        "kind": "top" if direction_index < config["axes"] else "random",
                        "direction": direction_index if direction_index < config["axes"] else direction_index - config["axes"],
                        "fidelity": fidelity,
                        "cosine": cosine,
                        "native_kl": exact_kl(native_logits, changed_logits[direction_index]),
                    }
                    if direction_index < config["axes"]:
                        centered = tangent_effects[direction_index] - (
                            native_probability * tangent_effects[direction_index]
                        ).sum()
                        probability_derivative = native_probability * centered
                        count = config["sign_tokens_per_side"]
                        indices = torch.cat((
                            torch.topk(probability_derivative, count).indices,
                            torch.topk(-probability_derivative, count).indices,
                        ))
                        finite_delta = F.softmax(changed_logits[direction_index].float(), -1) - native_probability
                        row["token_sign_agreement"] = float(
                            (torch.sign(finite_delta[indices]) == torch.sign(probability_derivative[indices])).float().mean()
                        )
                    scale_rows.append(row)
                site_rows.append({"scale": scale, "directions": scale_rows})
            records.append({"row": int(row_index), "scales": site_rows})
            print(corpus, site_number + 1, "/", len(sites), flush=True)

        summary = {}
        for scale in config["edit_norm_fractions"]:
            flat = [
                direction for site in records for scale_row in site["scales"]
                if scale_row["scale"] == scale for direction in scale_row["directions"]
            ]
            top = [row for row in flat if row["kind"] == "top"]
            random = [row for row in flat if row["kind"] == "random"]
            summary[str(scale)] = {
                "top_fidelity": median(top, "fidelity"),
                "top_cosine": median(top, "cosine"),
                "top_native_kl": median(top, "native_kl"),
                "random_native_kl": median(random, "native_kl"),
                "top_to_random_kl_ratio": median(top, "native_kl") / max(median(random, "native_kl"), 1e-30),
                "top_token_sign_agreement": median(top, "token_sign_agreement"),
            }
        results[corpus] = {"summary": summary, "records": records}

    def all_at(scale, key, threshold):
        return all(results[c]["summary"][str(scale)][key] >= threshold for c in results)

    hypotheses = {
        "D1_tangent_is_exact_at_point_one_percent": {"supported": all_at(0.001, "top_fidelity", 0.99)},
        "D2_tangent_remains_faithful_at_one_percent": {"supported": all_at(0.01, "top_fidelity", 0.95)},
        "D3_tangent_is_useful_at_five_percent": {"supported": all_at(0.05, "top_fidelity", 0.80)},
        "D4_axis_token_signs_validate": {"supported": all_at(0.01, "top_token_sign_agreement", 0.90)},
        "D5_top_axes_beat_random_directions": {"supported": all_at(0.01, "top_to_random_kl_ratio", 2.0)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Finite validation of local Layer-8 behavioral axes",
        "",
        "The basis-invariant Jacobian's four leading right-singular directions are edited directly in the normalized Layer-8 MLP input. Full nonlinear MLP and suffix computation is rerun. Seeded random directions in the same 512D state basis provide matched-norm controls. These are local off-manifold sensitivity tests, not claims that the edit is a natural language counterfactual.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Local axis validation](layer8_local_axis_validation.png)",
        "",
        "| Corpus | Edit | Fidelity / cosine | Top KL | Random KL | KL ratio | Token-sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for scale in config["edit_norm_fractions"]:
            row = result["summary"][str(scale)]
            lines.append(
                f"| {corpus} | {100 * scale:.1f}% | {row['top_fidelity']:.3f} / {row['top_cosine']:.3f} | "
                f"{row['top_native_kl']:.3e} | {row['random_native_kl']:.3e} | "
                f"{row['top_to_random_kl_ratio']:.1f}x | {row['top_token_sign_agreement']:.3f} |"
            )
    lines += [
        "",
        "The test distinguishes two questions: whether local axes are finite-causally real, and whether the same axes recur globally. The former can succeed even though the preceding recurrence audit failed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, result in results.items():
        scales = config["edit_norm_fractions"]
        axes[0].plot([100 * value for value in scales], [result["summary"][str(value)]["top_fidelity"] for value in scales], marker="o", label=corpus)
        axes[1].plot([100 * value for value in scales], [result["summary"][str(value)]["top_to_random_kl_ratio"] for value in scales], marker="o", label=corpus)
    axes[0].axhline(0.8, color="black", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Fisher fidelity")
    axes[0].set_title("Tangent to finite effect")
    axes[1].axhline(2, color="black", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("top/random KL")
    axes[1].set_title("Causal concentration")
    for axis in axes:
        axis.set_xlabel("input edit (% norm)")
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
