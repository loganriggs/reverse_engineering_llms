#!/usr/bin/env python3
"""Test whether the final-destination attention message is low-rank off the native-head basis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_final_destination_subspace_audit.json")
HEAD_SOURCE = TENSOR_ROOT / "bilin12_final_destination_head_program.json"
OUTPUT = TENSOR_ROOT / "bilin12_final_destination_subspace_audit.json"
ARRAYS = TENSOR_ROOT / "bilin12_final_destination_subspace_audit.npz"
REPORT = TENSOR_ROOT / "BILIN12_FINAL_DESTINATION_SUBSPACE_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_final_destination_subspace_audit.png"


def project(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (values @ basis) @ basis.T


def right_basis(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svd(values.float(), full_matrices=False).Vh.T.contiguous()


def fit_bases(values: dict[str, torch.Tensor], seed: int):
    layerwise = {layer: right_basis(matrix) for layer, matrix in values.items()}
    normalized = []
    for matrix in values.values():
        scale = matrix.square().sum(1).mean().sqrt().clamp_min(1e-12)
        normalized.append(matrix / scale)
    shared = right_basis(torch.cat(normalized))
    dimension = next(iter(values.values())).shape[1]
    random = {}
    for index, layer in enumerate(values):
        generator = torch.Generator().manual_seed(seed + index)
        random[layer] = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator), mode="reduced").Q
    return layerwise, shared, random


@torch.inference_mode()
def collect_deltas(model, tokenizer, rows, config: dict) -> dict[str, torch.Tensor]:
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    collected = {str(layer): [] for layer in config["layers"]}
    for edit_position in config["edit_positions"]:
        for start in range(0, len(rows), config["batch_size"]):
            raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
            base_ids = torch.as_tensor(raw, device="cuda", dtype=torch.long)
            keep = base_ids[:, edit_position] != newline
            if not bool(keep.any()): continue
            base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline
            base_rows, _, _ = trace(model, base_ids); changed_rows, _, _ = trace(model, changed_ids)
            for layer in config["layers"]:
                delta = changed_rows[layer]["details"]["write"][:, config["prediction_position"]] - base_rows[layer]["details"]["write"][:, config["prediction_position"]]
                collected[str(layer)].append(delta.float().cpu())
    return {layer: torch.cat(parts) for layer, parts in collected.items()}


def empty_metric() -> dict:
    return {"examples": 0, "target_energy": 0.0, "prediction_energy": 0.0, "cross": 0.0, "error": 0.0, "full_forward_kl": 0.0, "full_reverse_kl": 0.0, "forward_kl": 0.0, "target_error_kl": 0.0, "state_target": 0.0, "state_cross": 0.0, "state_error": 0.0}


def finish_metric(row: dict) -> dict:
    target = max(row["target_energy"], 1e-30); prediction = max(row["prediction_energy"], 1e-30); state = max(row["state_target"], 1e-30)
    return {
        "examples": row["examples"],
        "fisher_fidelity": 1 - row["error"] / target,
        "fisher_projection": row["cross"] / target,
        "fisher_cosine": row["cross"] / max(np.sqrt(target * prediction), 1e-30),
        "forward_kl_ratio": row["forward_kl"] / max(row["full_forward_kl"], 1e-30),
        "target_error_kl_ratio": row["target_error_kl"] / max(row["full_reverse_kl"], 1e-30),
        "state_fidelity": 1 - row["state_error"] / state,
        "state_projection": row["state_cross"] / state,
    }


@torch.inference_mode()
def evaluate_condition(model, tokenizer, rows, edit_position: int, bases: dict, config: dict) -> dict:
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    names = [(family, rank) for family in ("layerwise", "shared", "random") for rank in config["ranks"]]
    accumulators = {str(layer): {f"{family}/{rank}": empty_metric() for family, rank in names} for layer in config["layers"]}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda", dtype=torch.long); keep = base_ids[:, edit_position] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline
        base_rows, _, _ = trace(model, base_ids); changed_rows, _, _ = trace(model, changed_ids)
        for layer in config["layers"]:
            key = str(layer); base = base_rows[layer]; changed = changed_rows[layer]
            target_delta = changed["details"]["write"][:, config["prediction_position"]].float() - base["details"]["write"][:, config["prediction_position"]].float()
            full_write = base["details"]["write"].clone(); full_write[:, config["prediction_position"]] = changed["details"]["write"][:, config["prediction_position"]]
            endpoints = torch.stack((base["details"]["write"], full_write))
            endpoint_logits = logits_from_attention_writes(model, layer, base, endpoints)
            base_logits, target_logits = endpoint_logits; full_logit_delta = target_logits - base_logits
            target_energy, _, _, _ = fisher_terms(base_logits, full_logit_delta, full_logit_delta)
            full_forward = exact_kl(base_logits, target_logits).sum(); full_reverse = exact_kl(target_logits, base_logits).sum()
            worlds = []
            for family, rank in names:
                basis = bases[family][key][:, :rank].to(device="cuda", dtype=torch.float32)
                approximation = project(target_delta, basis)
                write = base["details"]["write"].clone(); write[:, config["prediction_position"]] += approximation.to(write.dtype)
                worlds.append(write)
            logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
            for index, (family, rank) in enumerate(names):
                row = accumulators[key][f"{family}/{rank}"]; approximation = project(target_delta, bases[family][key][:, :rank].to(device="cuda", dtype=torch.float32))
                prediction_delta = logits[index] - base_logits
                _, prediction_energy, cross, error = fisher_terms(base_logits, full_logit_delta, prediction_delta)
                row["examples"] += len(base_ids); row["target_energy"] += float(target_energy.sum()); row["prediction_energy"] += float(prediction_energy.sum()); row["cross"] += float(cross.sum()); row["error"] += float(error.sum())
                row["full_forward_kl"] += float(full_forward); row["full_reverse_kl"] += float(full_reverse); row["forward_kl"] += float(exact_kl(base_logits, logits[index]).sum()); row["target_error_kl"] += float(exact_kl(target_logits, logits[index]).sum())
                row["state_target"] += float(target_delta.square().sum()); row["state_cross"] += float((approximation * target_delta).sum()); row["state_error"] += float((approximation - target_delta).square().sum())
    return {layer: {name: finish_metric(row) for name, row in families.items()} for layer, families in accumulators.items()}


def spearman(first: list[float], second: list[float]) -> float:
    def ranks(values):
        order = np.argsort(values, kind="stable"); result = np.empty(len(values)); result[order] = np.arange(len(values)); return result
    return float(np.corrcoef(ranks(first), ranks(second))[0, 1])


def main() -> None:
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    selection_rows = load_rows(config["selection"]); print("collecting selection messages", flush=True); deltas = collect_deltas(model, tokenizer, selection_rows, config)
    layerwise, shared, random = fit_bases(deltas, config["seed"])
    max_rank = max(config["ranks"]); bases = {
        "layerwise": {layer: basis[:, :max_rank] for layer, basis in layerwise.items()},
        "shared": {layer: shared[:, :max_rank] for layer in layerwise},
        "random": {layer: basis[:, :max_rank] for layer, basis in random.items()},
    }
    arrays = {}
    for family, rows in bases.items():
        for layer, basis in rows.items(): arrays[f"{family}_layer{layer}"] = basis.numpy()
    for layer, values in deltas.items(): arrays[f"selection_delta_layer{layer}"] = values.numpy()
    np.savez_compressed(ARRAYS, **arrays)

    evaluation = {}
    for corpus, spec in config["evaluation"].items():
        rows = load_rows(spec); evaluation[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluating", corpus, position, flush=True)
            evaluation[corpus][str(position)] = evaluate_condition(model, tokenizer, rows, position, bases, config)

    def values(family: str, rank: int, metric: str = "fisher_fidelity") -> list[float]:
        return [evaluation[c][str(p)][str(layer)][f"{family}/{rank}"][metric] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    layer_rank16 = values("layerwise", 16); shared_rank16 = values("shared", 16); random_rank16 = values("random", 16)
    state_frontier = [float(np.mean(values("layerwise", rank, "state_fidelity"))) for rank in config["ranks"]]
    behavior_frontier = [float(np.mean(values("layerwise", rank))) for rank in config["ranks"]]
    transfer = {}
    for layer in config["layers"]:
        transfer[str(layer)] = {corpus: float(np.mean([evaluation[corpus][str(position)][str(layer)]["layerwise/16"]["fisher_fidelity"] for position in config["edit_positions"]])) for corpus in evaluation}
    head_source = json.loads(HEAD_SOURCE.read_text()); head_three = []
    for corpus, positions in head_source["evaluation"].items():
        for position, layers in positions.items():
            for layer, row in layers.items():
                mask = head_source["behavior_selected_masks"][layer]["3"]
                head_three.append(row["masks"][str(mask)]["fisher_fidelity"])
    hypotheses = {
        "P1_rank8_layerwise_basis_is_useful": {"supported": np.mean(values("layerwise", 8)) >= .75},
        "P2_rank16_layerwise_basis_is_nearly_sufficient": {"supported": min(layer_rank16) >= .85},
        "P3_rank32_layerwise_basis_is_robustly_sufficient": {"supported": min(values("layerwise", 32)) >= .95},
        "P4_one_shared_rank32_basis_is_nearly_sufficient": {"supported": min(values("shared", 32)) >= .85},
        "P5_learned_basis_beats_random": {"supported": np.mean(layer_rank16) - np.mean(random_rank16) >= .20, "mean_advantage": float(np.mean(layer_rank16) - np.mean(random_rank16))},
        "P6_state_variance_predicts_behavior_frontier": {"supported": spearman(state_frontier, behavior_frontier) >= .90, "spearman": spearman(state_frontier, behavior_frontier)},
        "P7_layerwise_orientation_matters": {"supported": np.mean(layer_rank16) - np.mean(shared_rank16) >= .05, "mean_advantage": float(np.mean(layer_rank16) - np.mean(shared_rank16))},
        "P8_rank16_layerwise_basis_transfers_ood": {"supported": all(abs(means[c] - means["fineweb"]) <= .15 for means in transfer.values() for c in ("wikitext", "tinystories")), "means": transfer},
        "P9_non_native_basis_beats_three_native_heads": {"supported": np.mean(layer_rank16) > np.mean(head_three), "rank16_mean": float(np.mean(layer_rank16)), "three_head_mean": float(np.mean(head_three))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "selection_examples": {layer: len(row) for layer, row in deltas.items()}, "state_frontier": dict(zip(map(str, config["ranks"]), state_frontier)), "behavior_frontier": dict(zip(map(str, config["ranks"]), behavior_frontier)), "evaluation": evaluation}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    ranks = config["ranks"]; fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for family in ("layerwise", "shared", "random"):
        axes[0].plot(ranks, [np.mean(values(family, rank)) for rank in ranks], marker="o", label=f"{family} mean")
        axes[0].plot(ranks, [min(values(family, rank)) for rank in ranks], marker="x", ls="--", label=f"{family} minimum")
    axes[0].set_title("Causal message-subspace frontier"); axes[0].set_xscale("log", base=2)
    axes[1].plot(state_frontier, behavior_frontier, marker="o")
    for rank, x, y in zip(ranks, state_frontier, behavior_frontier): axes[1].annotate(str(rank), (x, y), fontsize=7)
    axes[1].set_xlabel("mean state fidelity"); axes[1].set_ylabel("mean Fisher fidelity"); axes[1].set_title("State versus behavior")
    x = np.arange(len(config["layers"])); width = .18
    for index, corpus in enumerate(evaluation): axes[2].bar(x + (index - 1) * width, [transfer[str(layer)][corpus] for layer in config["layers"]], width, label=corpus)
    axes[2].set_xticks(x, [f"L{layer}" for layer in config["layers"]]); axes[2].set_title("Rank-16 transfer")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xlabel("rank"); axes[0].set_ylabel("Fisher fidelity"); axes[2].set_ylabel("mean Fisher fidelity")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = ["# Final-destination message-subspace audit", "", "The six-head native program was not sparse, so this preregistered follow-up asks whether its summed final-destination message is compact in a non-native residual-space basis. Uncentered state PCA is fit on FineWeb message deltas. Layer-specific bases, one normalized pooled basis shared by all four layers, and matched random bases are frozen before evaluation on disjoint FineWeb, WikiText, and TinyStories examples.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Message subspace](bilin12_final_destination_subspace_audit.png)", "", "## Held-out causal frontier", "", "| Rank | Layerwise mean / min | Shared mean / min | Random mean / min | State mean |", "|---:|---:|---:|---:|---:|"]
    for rank in ranks: lines.append(f"| {rank} | {np.mean(values('layerwise', rank)):.3f} / {min(values('layerwise', rank)):.3f} | {np.mean(values('shared', rank)):.3f} / {min(values('shared', rank)):.3f} | {np.mean(values('random', rank)):.3f} / {min(values('random', rank)):.3f} | {np.mean(values('layerwise', rank, 'state_fidelity')):.3f} |")
    lines += ["", "## Rank-16 transfer", "", "| Layer | FineWeb | WikiText | TinyStories |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]: lines.append(f"| {layer} | {transfer[str(layer)]['fineweb']:.3f} | {transfer[str(layer)]['wikitext']:.3f} | {transfer[str(layer)]['tinystories']:.3f} |")
    lines += ["", f"The frozen three-native-head program averaged {np.mean(head_three):.3f} Fisher fidelity; layerwise rank-16 PCA averaged {np.mean(layer_rank16):.3f}. A PCA basis is a descriptive coordinate system learned from message deltas, not a native module. Its success would establish compact message geometry; it would not establish neuron or head monosemanticity."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
