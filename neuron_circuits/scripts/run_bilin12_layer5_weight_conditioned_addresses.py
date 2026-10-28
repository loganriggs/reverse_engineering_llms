#!/usr/bin/env python3
"""Derive exact Layer-5 tensor addresses from weight-conditioned forms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_weight_conditioned_addresses.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_weight_conditioned_addresses.json"
ARRAYS = TENSOR_ROOT / "bilin12_layer5_weight_conditioned_addresses.npz"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_WEIGHT_CONDITIONED_ADDRESSES.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_weight_conditioned_addresses.png"


def output_conditioned_form(
    down: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    coefficients = down.T @ output
    form = left.T @ (coefficients[:, None] * right)
    return 0.5 * (form + form.T)


def restricted_operator(
    down: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    edit: torch.Tensor,
) -> torch.Tensor:
    left_edit = left @ edit
    right_edit = right @ edit
    return down @ (left_edit[:, None] * right + right_edit[:, None] * left)


def trace_clean(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - torch.trace(matrix) / len(matrix) * torch.eye(
        len(matrix), device=matrix.device, dtype=matrix.dtype
    )


def absolute_eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix)
    order = values.abs().argsort(descending=True)
    return values[order], vectors[:, order]


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-30)


def first_energy_rank(singular: torch.Tensor, threshold: float) -> int:
    cumulative = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-30)
    return int(torch.nonzero(cumulative >= threshold, as_tuple=False)[0]) + 1


def prediction_r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(
        1 - (target - prediction).square().sum()
        / target.square().sum().clamp_min(1e-30)
    )


def token_extremes(
    scores: torch.Tensor,
    tokenizer,
    count: int = 5,
) -> dict[str, list[str]]:
    positive = torch.topk(scores, count).indices.tolist()
    negative = torch.topk(-scores, count).indices.tolist()
    return {
        "positive": [tokenizer.decode([index]) for index in positive],
        "negative": [tokenizer.decode([index]) for index in negative],
    }


def derive_cleaned_addresses(
    form: torch.Tensor,
    square: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weighted = trace_clean(square @ form @ square)
    values, whitened = absolute_eigensystem(weighted)
    edits = torch.stack([
        normalized(square @ whitened[:, index]) for index in range(count)
    ], dim=1)
    return values[:count], whitened[:, :count], edits


def derive_raw_addresses(
    form: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = absolute_eigensystem(trace_clean(form))
    edits = torch.stack([normalized(vectors[:, index]) for index in range(count)], dim=1)
    return values[:count], edits


@torch.inference_mode()
def analyze_address(
    name: str,
    group: str,
    edit: torch.Tensor,
    origin_output: torch.Tensor,
    down: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    square: torch.Tensor,
    inverse: torch.Tensor,
    discovery_states: torch.Tensor,
    evaluations: dict[str, torch.Tensor],
    ranks: list[int],
    model,
    tokenizer,
) -> tuple[dict, dict[str, np.ndarray]]:
    operator = restricted_operator(down, left, right, edit)
    weighted = operator @ square
    output_basis, singular, context_vh = torch.linalg.svd(weighted, full_matrices=False)
    context_basis = context_vh.T
    energy = singular.square()
    output_role_fraction = float(
        (origin_output @ weighted).square().sum() / energy.sum().clamp_min(1e-30)
    )
    sample = discovery_states[0]
    native = down @ (
        (left @ edit) * (right @ sample)
        + (right @ edit) * (left @ sample)
    )
    formula = operator @ sample
    identity_error = float((native - formula).norm() / native.norm().clamp_min(1e-30))
    corpora = {}
    for corpus, states in evaluations.items():
        whitened = states @ inverse
        target = states @ operator.T
        coordinates = whitened @ context_basis
        rank_rows = []
        for rank in ranks:
            prediction = (
                coordinates[:, :rank] * singular[:rank]
            ) @ output_basis[:, :rank].T
            rank_rows.append({
                "rank": rank,
                "heldout_r2": prediction_r2(target, prediction),
                "discovery_energy_fraction": float(
                    energy[:rank].sum() / energy.sum().clamp_min(1e-30)
                ),
            })
        corpora[corpus] = {"rank_rows": rank_rows}
    embedding = model.embedding.weight.float()
    context_read = inverse @ context_basis[:, 0]
    row = {
        "name": name,
        "group": group,
        "identity_relative_error": identity_error,
        "k90": first_energy_rank(singular, 0.90),
        "k95": first_energy_rank(singular, 0.95),
        "effective_rank": float(energy.sum().square() / energy.square().sum().clamp_min(1e-30)),
        "output_role_fraction": output_role_fraction,
        "leading_address_tokens": token_extremes(embedding @ edit, tokenizer),
        "leading_context_tokens": token_extremes(embedding @ context_read, tokenizer),
        "leading_write_tokens": token_extremes(
            model.lm_head(output_basis[:, 0][None]).squeeze(0), tokenizer
        ),
        "corpora": corpora,
    }
    arrays = {
        f"{name}_edit": edit.cpu().numpy().astype(np.float32),
        f"{name}_output_basis": output_basis[:, :128].cpu().numpy().astype(np.float32),
        f"{name}_context_basis": context_basis[:, :128].cpu().numpy().astype(np.float32),
        f"{name}_singular": singular.cpu().numpy().astype(np.float32),
    }
    return row, arrays


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    _, output_vectors = output_map_eigensystem(down, left, right)

    print("collecting Layer-5 covariance states", flush=True)
    discovery_states, _ = collect_states_writes(
        model, config["layer"], config["discovery"],
        config["positions"], config["batch_size"],
    )
    square, inverse, covariance_metadata = ridge_covariance_transforms(
        discovery_states, config["covariance_ridge_fraction"]
    )
    evaluations = {}
    for corpus, spec in config["evaluation"].items():
        states, _ = collect_states_writes(
            model, config["layer"], spec, config["positions"], config["batch_size"]
        )
        evaluations[corpus] = states

    leading_output = output_vectors[:, 0]
    leading_form = output_conditioned_form(down, left, right, leading_output)
    clean_values, whitened_addresses, clean_edits = derive_cleaned_addresses(
        leading_form, square, config["leading_addresses"]
    )
    raw_values, raw_edits = derive_raw_addresses(
        leading_form, config["leading_addresses"]
    )

    positions_per_row = len(config["positions"])
    row_ids = torch.arange(len(discovery_states), device="cuda") // positions_per_row
    even = discovery_states[row_ids % 2 == 0]
    odd = discovery_states[row_ids % 2 == 1]
    even_square, _, _ = ridge_covariance_transforms(even, config["covariance_ridge_fraction"])
    odd_square, _, _ = ridge_covariance_transforms(odd, config["covariance_ridge_fraction"])
    _, even_whitened, even_edits = derive_cleaned_addresses(
        leading_form, even_square, config["leading_addresses"]
    )
    _, odd_whitened, odd_edits = derive_cleaned_addresses(
        leading_form, odd_square, config["leading_addresses"]
    )
    even_edit_basis = torch.linalg.qr(even_edits, mode="reduced").Q
    odd_edit_basis = torch.linalg.qr(odd_edits, mode="reduced").Q
    even_white_basis = torch.linalg.qr(even_whitened, mode="reduced").Q
    odd_white_basis = torch.linalg.qr(odd_whitened, mode="reduced").Q
    edit_canonical = torch.linalg.svdvals(even_edit_basis.T @ odd_edit_basis)
    whitened_canonical = torch.linalg.svdvals(even_white_basis.T @ odd_white_basis)

    address_specs: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
    for index in range(config["leading_addresses"]):
        address_specs.append((f"leading_clean_{index}", "leading_clean", clean_edits[:, index], leading_output))
        address_specs.append((f"leading_raw_{index}", "leading_raw", raw_edits[:, index], leading_output))
    for mode in config["nonleading_output_modes"]:
        output = output_vectors[:, mode]
        form = output_conditioned_form(down, left, right, output)
        _, _, edits = derive_cleaned_addresses(form, square, 1)
        address_specs.append((f"nonleading_mode{mode}", "nonleading", edits[:, 0], output))

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    channel_coefficients = down.T @ leading_output
    for index in range(config["sign_nulls"]):
        signs = torch.where(
            torch.rand(len(channel_coefficients), generator=generator, device="cuda") < 0.5,
            -torch.ones((), device="cuda"), torch.ones((), device="cuda"),
        )
        null_form = left.T @ ((channel_coefficients * signs)[:, None] * right)
        null_form = 0.5 * (null_form + null_form.T)
        _, _, edits = derive_cleaned_addresses(null_form, square, 1)
        address_specs.append((f"sign_null_{index}", "sign_null", edits[:, 0], leading_output))
    for index in range(config["random_addresses"]):
        whitened = normalized(torch.randn(
            model.config.n_embd, generator=generator, device="cuda"
        ))
        edit = normalized(square @ whitened)
        address_specs.append((f"random_{index}", "random", edit, leading_output))

    results = {}
    arrays = {
        "covariance_square": square.cpu().numpy().astype(np.float32),
        "covariance_inverse": inverse.cpu().numpy().astype(np.float32),
        "leading_output": leading_output.cpu().numpy().astype(np.float32),
        "leading_clean_values": clean_values.cpu().numpy().astype(np.float32),
        "leading_raw_values": raw_values.cpu().numpy().astype(np.float32),
        "leading_whitened_addresses": whitened_addresses.cpu().numpy().astype(np.float32),
        "leading_clean_edits": clean_edits.cpu().numpy().astype(np.float32),
        "leading_raw_edits": raw_edits.cpu().numpy().astype(np.float32),
    }
    for name, group, edit, origin_output in address_specs:
        print("restricted address", name, flush=True)
        row, saved = analyze_address(
            name, group, edit, origin_output, down, left, right,
            square, inverse, discovery_states, evaluations, config["ranks"],
            model, tokenizer,
        )
        results[name] = row
        arrays.update(saved)

    groups: dict[str, list[dict]] = {}
    for row in results.values():
        groups.setdefault(row["group"], []).append(row)
    leading = groups["leading_clean"]
    raw = groups["leading_raw"]
    random_rows = groups["random"]
    sign_rows = groups["sign_null"]
    nonleading = groups["nonleading"]

    def rank64(row: dict, corpus: str) -> float:
        return next(
            item["heldout_r2"] for item in row["corpora"][corpus]["rank_rows"]
            if item["rank"] == 64
        )

    hypotheses = {
        "A1_restricted_derivative_identity_is_exact": {
            "supported": bool(max(row["identity_relative_error"] for row in results.values()) <= 1e-5),
            "maximum_relative_error": max(row["identity_relative_error"] for row in results.values()),
        },
        "A2_leading_addresses_are_rank64_compact": {
            "supported": bool(all(rank64(row, corpus) >= 0.80 for row in leading for corpus in evaluations)),
        },
        "A3_leading_addresses_beat_random_compactness": {
            "supported": bool(np.median([row["k90"] for row in leading])
            <= 0.75 * np.median([row["k90"] for row in random_rows])),
        },
        "A4_leading_addresses_have_clean_output_role": {
            "supported": bool(np.median([row["output_role_fraction"] for row in leading]) >= 0.50),
        },
        "A5_output_role_beats_sign_nulls": {
            "supported": bool(np.median([row["output_role_fraction"] for row in leading])
            > np.quantile([row["output_role_fraction"] for row in sign_rows], 0.95)),
        },
        "A6_rank64_compactness_transfers_ood": {
            "supported": bool(all(
                np.median([rank64(row, corpus) for row in leading]) >= 0.80
                for corpus in evaluations
            )),
        },
        "A7_covariance_cleaning_shortens_addresses": {
            "supported": bool(np.median([row["k90"] for row in leading])
            < np.median([row["k90"] for row in raw])),
        },
        "A8_cleaned_address_subspace_is_split_stable": {
            "supported": bool(float(edit_canonical.mean()) >= 0.80),
            "edit_mean_canonical_cosine": float(edit_canonical.mean()),
            "whitened_mean_canonical_cosine": float(whitened_canonical.mean()),
        },
        "A9_leading_mode_is_more_compact_than_nonleading_modes": {
            "supported": bool(np.median([row["k90"] for row in leading])
            < np.median([row["k90"] for row in nonleading])),
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "covariance": covariance_metadata,
        "hypotheses": hypotheses,
        "split_stability": {
            "edit_canonical_cosines": edit_canonical.cpu().tolist(),
            "whitened_canonical_cosines": whitened_canonical.cpu().tolist(),
        },
        "group_summary": {
            group: {
                "count": len(rows),
                "median_k90": float(np.median([row["k90"] for row in rows])),
                "median_output_role_fraction": float(np.median([row["output_role_fraction"] for row in rows])),
            }
            for group, rows in groups.items()
        },
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Independent Layer-5 weight-conditioned spectral addresses",
        "",
        "The leading output direction is selected from the complete bilinear weight tensor. Contracting that output direction into the tensor gives an exact symmetric input form. The top covariance-cleaned eigenvectors of that form define four input addresses without labels or clustering. Contracting each address into the native tensor gives an exact `context → write` derivative operator, whose frozen natural-metric SVD is evaluated on FineWeb, WikiText, and TinyStories. Raw-form, nonleading-output, sign-scrambled-channel, and random natural-direction controls are matched in the same pipeline.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-5 weight-conditioned addresses](bilin12_layer5_weight_conditioned_addresses.png)",
        "",
        f"Even/odd cleaned-address mean canonical cosine: **{float(edit_canonical.mean()):.3f}** in native edit space and **{float(whitened_canonical.mean()):.3f}** in whitened coordinates.",
        "",
        "| Address | K90 | Effective rank | Leading-output energy | Rank-64 R² FineWeb / WikiText / TinyStories |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in leading + raw + nonleading:
        values = " / ".join(f"{rank64(row, corpus):.3f}" for corpus in ("fineweb", "wikitext", "tinystories"))
        lines.append(
            f"| {row['name']} | {row['k90']} | {row['effective_rank']:.1f} | "
            f"{row['output_role_fraction']:.3f} | {values} |"
        )
    lines += [
        "",
        "| Group | Count | Median K90 | Median leading-output energy |",
        "|---|---:|---:|---:|",
    ]
    for group, rows in groups.items():
        lines.append(
            f"| {group} | {len(rows)} | {np.median([row['k90'] for row in rows]):.1f} | "
            f"{np.median([row['output_role_fraction'] for row in rows]):.3f} |"
        )
    lines += [
        "",
        "An address is an input direction that strongly changes one selected output score. It is not itself a complete semantic feature. Compact restricted rank would mean the address invokes a short reusable context-conditioned operation; high output-role energy would mean that operation mostly preserves the output question used to discover it. Sign and random controls test whether either property is specific to the apparent spectral spike.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for row in leading:
        axes[0].plot(
            config["ranks"],
            [item["heldout_r2"] for item in row["corpora"]["fineweb"]["rank_rows"]],
            marker="o", label=row["name"],
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(config["ranks"], config["ranks"], rotation=35)
    axes[0].set_title("Leading-address derivative frontier")
    axes[0].set_ylabel("FineWeb held-out R²")
    axes[0].legend(fontsize=7)
    group_order = ("leading_clean", "leading_raw", "nonleading", "sign_null", "random")
    axes[1].boxplot(
        [[row["k90"] for row in groups[group]] for group in group_order],
        tick_labels=group_order,
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("K90")
    axes[1].set_title("Restricted-operator rank")
    axes[2].boxplot(
        [[row["output_role_fraction"] for row in groups[group]] for group in group_order],
        tick_labels=group_order,
    )
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylabel("derivative energy in origin output")
    axes[2].set_title("Output-role preservation")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
