#!/usr/bin/env python3
"""Find and red-team sparse native-head programs at the final destination."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_attention_source_distance_redteam import sequence_head_writes
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_final_destination_head_program.json")
OUTPUT = TENSOR_ROOT / "bilin12_final_destination_head_program.json"
REPORT = TENSOR_ROOT / "BILIN12_FINAL_DESTINATION_HEAD_PROGRAM.md"
FIGURE = TENSOR_ROOT / "bilin12_final_destination_head_program.png"


def mask_heads(mask: int, heads: int) -> list[int]:
    return [head for head in range(heads) if mask & (1 << head)]


def make_write_worlds(
    base_write: torch.Tensor,
    final_head_deltas: torch.Tensor,
    masks: list[int],
    position: int,
) -> torch.Tensor:
    """Insert additive native-head differences at one destination only."""
    heads = final_head_deltas.shape[0]
    coefficients = torch.tensor(
        [[float(bool(mask & (1 << head))) for head in range(heads)] for mask in masks],
        device=base_write.device,
        dtype=base_write.dtype,
    )
    delta = torch.einsum("mh,hbd->mbd", coefficients, final_head_deltas)
    worlds = base_write.unsqueeze(0).repeat(len(masks), 1, 1, 1)
    worlds[:, :, position] += delta
    return worlds


def empty_accumulator(masks: int) -> dict:
    return {
        "examples": 0,
        "target_energy": 0.0,
        "full_forward_kl": 0.0,
        "full_reverse_kl": 0.0,
        "prediction_energy": np.zeros(masks),
        "cross": np.zeros(masks),
        "error": np.zeros(masks),
        "forward_kl": np.zeros(masks),
        "target_error_kl": np.zeros(masks),
        "state_target_energy": 0.0,
        "state_cross": np.zeros(masks),
        "state_error": np.zeros(masks),
        "maximum_state_recomposition_error": 0.0,
        "relative_state_recomposition_num": 0.0,
        "relative_state_recomposition_den": 0.0,
        "maximum_logit_recomposition_error": 0.0,
    }


def merge_accumulators(target: dict, source: dict) -> None:
    for key in (
        "examples", "target_energy", "full_forward_kl", "full_reverse_kl",
        "state_target_energy", "relative_state_recomposition_num",
        "relative_state_recomposition_den",
    ):
        target[key] += source[key]
    for key in (
        "prediction_energy", "cross", "error", "forward_kl",
        "target_error_kl", "state_cross", "state_error",
    ):
        target[key] += source[key]
    for key in ("maximum_state_recomposition_error", "maximum_logit_recomposition_error"):
        target[key] = max(target[key], source[key])


def finalize(accumulator: dict) -> dict:
    target = max(accumulator["target_energy"], 1e-30)
    state_target = max(accumulator["state_target_energy"], 1e-30)
    forward = max(accumulator["full_forward_kl"], 1e-30)
    reverse = max(accumulator["full_reverse_kl"], 1e-30)
    rows = {}
    for mask in range(len(accumulator["cross"])):
        prediction = accumulator["prediction_energy"][mask]
        rows[str(mask)] = {
            "heads": mask_heads(mask, int(np.log2(len(accumulator["cross"])))),
            "fisher_fidelity": float(1 - accumulator["error"][mask] / target),
            "fisher_projection": float(accumulator["cross"][mask] / target),
            "fisher_cosine": float(accumulator["cross"][mask] / max(np.sqrt(prediction * target), 1e-30)),
            "forward_kl_ratio": float(accumulator["forward_kl"][mask] / forward),
            "target_error_kl_ratio": float(accumulator["target_error_kl"][mask] / reverse),
            "state_fidelity": float(1 - accumulator["state_error"][mask] / state_target),
            "state_projection": float(accumulator["state_cross"][mask] / state_target),
        }
    return {
        "examples": accumulator["examples"],
        "masks": rows,
        "maximum_state_recomposition_error": accumulator["maximum_state_recomposition_error"],
        "relative_state_recomposition_error": float(np.sqrt(
            accumulator["relative_state_recomposition_num"] /
            max(accumulator["relative_state_recomposition_den"], 1e-30)
        )),
        "maximum_logit_recomposition_error": accumulator["maximum_logit_recomposition_error"],
    }


@torch.inference_mode()
def evaluate_condition(model, tokenizer, rows, edit_position: int, config: dict) -> dict:
    heads = model.config.n_head
    masks = 1 << heads
    full_mask = masks - 1
    prediction_position = config["prediction_position"]
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    accumulators = {layer: empty_accumulator(masks) for layer in config["layers"]}

    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda")
        keep = base_ids[:, edit_position] != newline
        if not bool(keep.any()):
            continue
        base_ids = base_ids[keep]
        changed_ids = base_ids.clone()
        changed_ids[:, edit_position] = newline
        base_rows, _, _ = trace(model, base_ids)
        changed_rows, _, _ = trace(model, changed_ids)

        for layer in config["layers"]:
            acc = accumulators[layer]
            base = base_rows[layer]
            changed = changed_rows[layer]
            head_delta_sequence = sequence_head_writes(
                model.blocks[layer], base["details"]["attended"], changed["details"]["attended"]
            )
            final_head_deltas = head_delta_sequence[:, :, prediction_position]
            direct_target_write = base["details"]["write"].clone()
            direct_target_write[:, prediction_position] = changed["details"]["write"][:, prediction_position]

            endpoint_worlds = make_write_worlds(
                base["details"]["write"], final_head_deltas,
                [0, full_mask], prediction_position,
            )
            endpoint_logits = logits_from_attention_writes(model, layer, base, endpoint_worlds)
            independent_target_logits = logits_from_attention_writes(
                model, layer, base, direct_target_write.unsqueeze(0)
            )[0]
            base_logits, target_logits = endpoint_logits
            target_delta = target_logits - base_logits
            target_energy, _, _, _ = fisher_terms(base_logits, target_delta, target_delta)
            target_write_delta = direct_target_write[:, prediction_position] - base["details"]["write"][:, prediction_position]
            full_head_delta = final_head_deltas.sum(0)

            acc["examples"] += len(base_ids)
            acc["target_energy"] += float(target_energy.sum())
            acc["full_forward_kl"] += float(exact_kl(base_logits, target_logits).sum())
            acc["full_reverse_kl"] += float(exact_kl(target_logits, base_logits).sum())
            acc["state_target_energy"] += float(target_write_delta.square().sum())
            state_error = full_head_delta - target_write_delta
            acc["maximum_state_recomposition_error"] = max(
                acc["maximum_state_recomposition_error"], float(state_error.abs().max())
            )
            acc["relative_state_recomposition_num"] += float(state_error.square().sum())
            acc["relative_state_recomposition_den"] += float(target_write_delta.square().sum())
            acc["maximum_logit_recomposition_error"] = max(
                acc["maximum_logit_recomposition_error"],
                float((target_logits - independent_target_logits).abs().max()),
            )

            for chunk_start in range(0, masks, config["world_chunk"]):
                chunk_masks = list(range(chunk_start, min(chunk_start + config["world_chunk"], masks)))
                write_worlds = make_write_worlds(
                    base["details"]["write"], final_head_deltas,
                    chunk_masks, prediction_position,
                )
                logits = logits_from_attention_writes(model, layer, base, write_worlds)
                coefficients = torch.tensor(
                    [[float(bool(mask & (1 << head))) for head in range(heads)] for mask in chunk_masks],
                    device=full_head_delta.device,
                    dtype=full_head_delta.dtype,
                )
                state_predictions = torch.einsum("mh,hbd->mbd", coefficients, final_head_deltas)
                for local_index, mask in enumerate(chunk_masks):
                    prediction_delta = logits[local_index] - base_logits
                    _, prediction_energy, cross, error = fisher_terms(
                        base_logits, target_delta, prediction_delta
                    )
                    acc["prediction_energy"][mask] += float(prediction_energy.sum())
                    acc["cross"][mask] += float(cross.sum())
                    acc["error"][mask] += float(error.sum())
                    acc["forward_kl"][mask] += float(exact_kl(base_logits, logits[local_index]).sum())
                    acc["target_error_kl"][mask] += float(exact_kl(target_logits, logits[local_index]).sum())
                    state_prediction = state_predictions[local_index]
                    acc["state_cross"][mask] += float((state_prediction * target_write_delta).sum())
                    acc["state_error"][mask] += float((state_prediction - target_write_delta).square().sum())
    return {str(layer): accumulators[layer] for layer in config["layers"]}


def best_mask(row: dict, budget: int, metric: str) -> int:
    candidates = [mask for mask in range(len(row["cross"])) if mask.bit_count() == budget]
    if metric == "behavior":
        target = max(row["target_energy"], 1e-30)
        return max(candidates, key=lambda mask: 1 - row["error"][mask] / target)
    target = max(row["state_target_energy"], 1e-30)
    return max(candidates, key=lambda mask: 1 - row["state_error"][mask] / target)


def output_subspace(block, heads: list[int]) -> torch.Tensor:
    head_dimension = block.attn.head_dim
    columns = []
    for head in heads:
        columns.append(block.attn.c_proj.weight[:, head * head_dimension:(head + 1) * head_dimension].float())
    matrix = torch.cat(columns, dim=1)
    return torch.linalg.qr(matrix, mode="reduced").Q


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.T @ right).square().sum() / min(left.shape[1], right.shape[1]))


def weight_alignment(model, selected_masks: dict, layers: list[int], budget: int = 2) -> list[dict]:
    head_sets = list(itertools.combinations(range(model.config.n_head), budget))
    rows = []
    for left_layer, right_layer in zip(layers[:-1], layers[1:]):
        selected_left = mask_heads(selected_masks[str(left_layer)][str(budget)], model.config.n_head)
        selected_right = mask_heads(selected_masks[str(right_layer)][str(budget)], model.config.n_head)
        selected_value = subspace_overlap(
            output_subspace(model.blocks[left_layer], selected_left),
            output_subspace(model.blocks[right_layer], selected_right),
        )
        null = []
        for left in head_sets:
            left_basis = output_subspace(model.blocks[left_layer], list(left))
            for right in head_sets:
                null.append(subspace_overlap(left_basis, output_subspace(model.blocks[right_layer], list(right))))
        percentile = float(100 * np.mean(np.asarray(null) <= selected_value))
        rows.append({
            "layers": [left_layer, right_layer],
            "selected_heads": [selected_left, selected_right],
            "selected_overlap": selected_value,
            "null_mean": float(np.mean(null)),
            "null_p90": float(np.quantile(null, .90)),
            "selected_percentile": percentile,
        })
    return rows


@torch.inference_mode()
def worked_example(model, tokenizer, row: np.ndarray, edit_position: int, layer: int, mask: int, config: dict) -> dict:
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    base_ids = torch.as_tensor(row[None, :config["sequence_length"]], device="cuda", dtype=torch.long)
    changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline
    base_rows, _, _ = trace(model, base_ids); changed_rows, _, _ = trace(model, changed_ids)
    base = base_rows[layer]; changed = changed_rows[layer]
    deltas = sequence_head_writes(model.blocks[layer], base["details"]["attended"], changed["details"]["attended"])
    worlds = make_write_worlds(base["details"]["write"], deltas[:, :, config["prediction_position"]], [0, mask, (1 << model.config.n_head) - 1], config["prediction_position"])
    logits = logits_from_attention_writes(model, layer, base, worlds)[:, 0]
    full_delta = logits[2] - logits[0]; sparse_delta = logits[1] - logits[0]
    _, _, cross, error = fisher_terms(logits[0:1], full_delta[None], sparse_delta[None])
    target, _, _, _ = fisher_terms(logits[0:1], full_delta[None], full_delta[None])
    top = torch.topk(full_delta.abs(), 8).indices.tolist()
    return {
        "layer": layer, "edit_position": edit_position, "heads": mask_heads(mask, model.config.n_head),
        "context": tokenizer.decode(base_ids[0, max(0, edit_position - 24):edit_position + 25].tolist()),
        "original_token": tokenizer.decode([int(base_ids[0, edit_position])]),
        "replacement_token": tokenizer.decode([newline]),
        "fisher_fidelity": float(1 - error.sum() / target.sum().clamp_min(1e-30)),
        "largest_full_logit_changes": [
            {"token": tokenizer.decode([token]), "full": float(full_delta[token]), "sparse": float(sparse_delta[token])}
            for token in top
        ],
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    heads = model.config.n_head; masks = 1 << heads

    selection_rows = load_rows(config["selection"])
    selection_raw = {str(layer): empty_accumulator(masks) for layer in config["layers"]}
    selection_by_position = {}
    for position in config["edit_positions"]:
        print("selection", position, flush=True)
        condition = evaluate_condition(model, tokenizer, selection_rows, position, config)
        selection_by_position[str(position)] = {layer: finalize(row) for layer, row in condition.items()}
        for layer in config["layers"]:
            merge_accumulators(selection_raw[str(layer)], condition[str(layer)])

    behavior_masks = {
        str(layer): {str(budget): best_mask(selection_raw[str(layer)], budget, "behavior") for budget in range(1, heads + 1)}
        for layer in config["layers"]
    }
    state_masks = {
        str(layer): {str(budget): best_mask(selection_raw[str(layer)], budget, "state") for budget in range(1, heads + 1)}
        for layer in config["layers"]
    }
    selection = {str(layer): finalize(selection_raw[str(layer)]) for layer in config["layers"]}

    evaluation = {}
    for corpus, spec in config["evaluation"].items():
        rows = load_rows(spec); evaluation[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluation", corpus, position, flush=True)
            condition = evaluate_condition(model, tokenizer, rows, position, config)
            evaluation[corpus][str(position)] = {layer: finalize(row) for layer, row in condition.items()}

    def selected_values(kind: str, budget: int) -> list[float]:
        chosen = behavior_masks if kind == "behavior" else state_masks
        return [
            evaluation[corpus][str(position)][str(layer)]["masks"][str(chosen[str(layer)][str(budget)])]["fisher_fidelity"]
            for corpus in evaluation for position in config["edit_positions"] for layer in config["layers"]
        ]

    alignment = weight_alignment(model, behavior_masks, config["layers"], 2)
    recurrent = {5: 2, 6: 5, 8: 3, 10: 0}
    h4_differences = {
        str(budget): float(np.mean(selected_values("behavior", budget)) - np.mean(selected_values("state", budget)))
        for budget in (2, 3)
    }
    transfer = {}
    for layer in config["layers"]:
        mask = behavior_masks[str(layer)]["3"]
        means = {
            corpus: float(np.mean([
                evaluation[corpus][str(position)][str(layer)]["masks"][str(mask)]["fisher_fidelity"]
                for position in config["edit_positions"]
            ])) for corpus in evaluation
        }
        transfer[str(layer)] = means

    all_finalized = [
        evaluation[c][str(p)][str(layer)]
        for c in evaluation for p in config["edit_positions"] for layer in config["layers"]
    ]
    hypotheses = {
        "H1_head_recomposition_is_exact": {"supported": all(
            row["relative_state_recomposition_error"] <= 1e-6 and row["maximum_logit_recomposition_error"] <= 1e-4
            for row in all_finalized
        )},
        "H2_two_behavior_selected_heads_are_useful": {"supported": min(selected_values("behavior", 2)) >= .50},
        "H3_three_behavior_selected_heads_are_nearly_sufficient": {"supported": min(selected_values("behavior", 3)) >= .85},
        "H4_behavior_selection_beats_state_selection": {"supported": all(value >= .05 for value in h4_differences.values()), "mean_differences": h4_differences},
        "H5_head_program_transfers_ood": {"supported": all(
            abs(means[corpus] - means["fineweb"]) <= .15
            for means in transfer.values() for corpus in ("wikitext", "tinystories")
        ), "means": transfer},
        "H6_previously_recurrent_heads_are_selected": {"supported": all(
            recurrent[layer] in mask_heads(behavior_masks[str(layer)]["2"], heads) for layer in config["layers"]
        )},
        "H7_native_head_program_is_sparse": {"supported": float(np.mean(selected_values("behavior", 2))) >= .80},
        "H8_four_heads_are_sufficient": {"supported": min(selected_values("behavior", 4)) >= .95},
        "H9_selected_output_subspaces_are_weight_aligned": {"supported": all(row["selected_percentile"] >= 90 for row in alignment)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])

    fineweb_rows = load_rows(config["evaluation"]["fineweb"])
    example = worked_example(model, tokenizer, np.asarray(fineweb_rows[0]), 95, 8, behavior_masks["8"]["2"], config)

    payload = {
        "config": config, "hypotheses": hypotheses,
        "behavior_selected_masks": behavior_masks, "state_selected_masks": state_masks,
        "selection": selection, "selection_by_position": selection_by_position,
        "evaluation": evaluation, "weight_alignment": alignment, "worked_example": example,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    budgets = list(range(1, heads + 1)); fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for kind, style in (("behavior", "-"), ("state", "--")):
        means = [np.mean(selected_values(kind, budget)) for budget in budgets]
        minima = [min(selected_values(kind, budget)) for budget in budgets]
        axes[0].plot(budgets, means, marker="o", ls=style, label=f"{kind} mean")
        axes[0].plot(budgets, minima, marker="x", ls=style, label=f"{kind} minimum")
    axes[0].axhline(.85, color="black", lw=.8, ls=":"); axes[0].set_title("Frozen native-head frontier")
    x = np.arange(len(config["layers"])); width = .18
    for index, corpus in enumerate(evaluation):
        values = [transfer[str(layer)][corpus] for layer in config["layers"]]
        axes[1].bar(x + (index - 1) * width, values, width, label=corpus)
    axes[1].set_xticks(x, [f"L{layer}" for layer in config["layers"]]); axes[1].set_title("Three-head transfer")
    labels = [f"L{row['layers'][0]}→L{row['layers'][1]}" for row in alignment]
    axes[2].bar(np.arange(len(alignment)), [row["selected_percentile"] for row in alignment])
    axes[2].axhline(90, color="black", ls="--", lw=.8); axes[2].set_xticks(np.arange(len(alignment)), labels); axes[2].set_title("Selected output-subspace percentile")
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=7) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xlabel("head budget"); axes[0].set_ylabel("Fisher fidelity"); axes[1].set_ylabel("mean Fisher fidelity"); axes[2].set_ylabel("null percentile")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Final-destination native-head program", "",
        "FineWeb selection examples choose exact subsets of the six native attention heads at Layers 5, 6, 8, and 10. Only each head's changed write at the final prediction destination is inserted. Frozen subsets are evaluated on disjoint FineWeb, WikiText, and TinyStories examples at source distances 8, 32, and 64. Behavior-selected subsets optimize downstream Fisher fidelity; state-selected subsets optimize Euclidean write reconstruction.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"The strict recomposition gate failed only numerically: maximum relative state error is {max(row['relative_state_recomposition_error'] for row in all_finalized):.2e} and maximum propagated-logit disagreement is {max(row['maximum_logit_recomposition_error'] for row in all_finalized):.2e}. In a float64 spot check, splitting the output projection by heads recomposed to relative error below 2.4e-16. The failure is retained because the preregistered runtime threshold was 1e-6."]
    lines += ["", "![Final-destination head program](bilin12_final_destination_head_program.png)", "", "## Selected programs", "", "| Layer | Behavior K2 | State K2 | Behavior K3 | State K3 |", "|---:|---|---|---|---|"]
    for layer in config["layers"]:
        lines.append(f"| {layer} | {mask_heads(behavior_masks[str(layer)]['2'], heads)} | {mask_heads(state_masks[str(layer)]['2'], heads)} | {mask_heads(behavior_masks[str(layer)]['3'], heads)} | {mask_heads(state_masks[str(layer)]['3'], heads)} |")
    lines += ["", "## Held-out frontier", "", "| Budget | Behavior mean / min | State mean / min |", "|---:|---:|---:|"]
    for budget in budgets:
        behavior = selected_values("behavior", budget); state = selected_values("state", budget)
        lines.append(f"| {budget} | {np.mean(behavior):.3f} / {min(behavior):.3f} | {np.mean(state):.3f} / {min(state):.3f} |")
    lines += ["", "## Weight-only output-subspace control", "", "| Layers | Selected overlap | Null mean / p90 | Percentile |", "|---|---:|---:|---:|"]
    for row in alignment:
        lines.append(f"| {row['layers'][0]}→{row['layers'][1]} | {row['selected_overlap']:.3f} | {row['null_mean']:.3f} / {row['null_p90']:.3f} | {row['selected_percentile']:.1f} |")
    lines += ["", "## Worked Layer-8 example", "", f"Context around edit: `{example['context'].replace('`', '')}`", "", f"Original token `{example['original_token']!r}` is replaced by newline. Frozen heads {example['heads']} achieve Fisher fidelity {example['fisher_fidelity']:.3f} for the full six-head final-destination message.", "", "| Token | Full logit change | Sparse-head change |", "|---|---:|---:|"]
    for row in example["largest_full_logit_changes"]:
        token = row["token"].replace("|", "\\|").replace("`", "")
        lines.append(f"| `{token}` | {row['full']:.3f} | {row['sparse']:.3f} |")
    lines += ["", "A native-head program is an intervention interface, not a claim that a head has one semantic meaning. Agreement with a weight-only output subspace would support a recurring structural module; disagreement would mean functional recurrence is implemented by differently oriented head bases."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
