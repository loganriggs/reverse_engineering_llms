#!/usr/bin/env python3
"""Blindly forecast sitewise errors of the sparse graph replacement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_graph_natural_causality import final_components, kl, logits
from run_bilin12_router_interaction_atlas import fast_auc
from run_bilin12_router_joint_diagonalization import evaluate_forms, masked_forms
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_graph_blind_prediction.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_graph_blind_prediction.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_GRAPH_BLIND_PREDICTION.md"


def ranks(values):
    order = np.argsort(values, kind="mergesort"); result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values)); return result


def spearman(left, right):
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def rmsnorm_jvp(hidden, tangent):
    epsilon = torch.finfo(hidden.dtype).eps
    mean_square = hidden.square().mean(-1, keepdim=True) + epsilon
    inverse = mean_square.rsqrt()
    return tangent * inverse - hidden * (hidden * tangent).mean(-1, keepdim=True) * inverse.pow(3)


def logit_jvp(model, hidden, tangent):
    normalized = F.rms_norm(hidden, (hidden.shape[-1],))
    raw = model.lm_head(normalized).float()
    raw_tangent = model.lm_head(rmsnorm_jvp(hidden.float(), tangent.float())).float()
    return raw_tangent * (1 - torch.tanh(raw / 30).square())


def summarize(native, finite, predicted, hidden_delta):
    probability = F.softmax(native.float(), -1)
    finite_delta = finite.float() - native.float(); predicted_delta = predicted.float()
    def center(value): return value - (probability * value).sum(-1, keepdim=True)
    finite_centered = center(finite_delta); predicted_centered = center(predicted_delta)
    finite_energy = (probability * finite_centered.square()).sum(-1)
    predicted_energy = (probability * predicted_centered.square()).sum(-1)
    cross = (probability * finite_centered * predicted_centered).sum(-1)
    cosine = cross / (finite_energy * predicted_energy).sqrt().clamp_min(1e-30)
    actual_kl = kl(native, finite)
    predicted_kl = .5 * predicted_energy
    top = native.argmax(-1)
    predicted_logits = native + predicted_delta
    top_score = predicted_logits.gather(-1, top[:, None]).squeeze(-1)
    masked = predicted_logits.clone(); masked.scatter_(-1, top[:, None], -torch.inf)
    flip_score = masked.max(-1).values - top_score
    actual_flip = finite.argmax(-1) != top
    return {"cosine": cosine, "actual_kl": actual_kl, "predicted_kl": predicted_kl,
            "hidden_norm": hidden_delta.norm(dim=-1), "flip_score": flip_score,
            "actual_flip": actual_flip}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); parent_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(config["parent_result"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(parent_config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**parent_config, "rank": parent_config["output_rank"],
                                                   "discovery": parent_config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, parent_config["natural"], parent_config["natural_positions"],
                                       parent_config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False); natural_basis = vh[:parent_config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, parent_config); rotation = rotation.detach(); forms = rotation.T[None] @ restricted @ rotation[None]
    selected = [tuple(pair) for pair in parent["selected_pairs"]]
    selected_forms = masked_forms(forms, selected, len(selected))
    positions = np.asarray(parent_config["positions"], dtype=np.int64)
    results = {}
    with torch.inference_mode():
        for corpus, spec in config["evaluation"].items():
            print("evaluate", corpus, flush=True)
            source = np.load(spec["path"], mmap_mode="r"); rows = source[spec["start"]:spec["start"] + spec["rows"]]
            collected = {name: [] for name in ("cosine", "actual_kl", "predicted_kl", "hidden_norm", "flip_score", "actual_flip")}
            for start in range(0, len(rows), parent_config["batch_size"]):
                raw_ids = np.asarray(rows[start:start + parent_config["batch_size"]], dtype=np.int64)
                ids = torch.as_tensor(raw_ids[:, :128], device="cuda")
                pre, state, write, native = final_components(model, ids, positions)
                coordinates_value = state.flatten(0, 1) @ natural_basis @ rotation
                exact_coefficients = ((write - block.mlp.Down_bias) @ write_basis).flatten(0, 1)
                selected_coefficients = evaluate_forms(selected_forms, torch.zeros_like(coordinates_value), coordinates_value)
                hidden_delta = ((selected_coefficients - exact_coefficients) @ write_basis.T).reshape_as(write)
                finite = logits(model, pre + write + hidden_delta)
                predicted_delta = logit_jvp(model, pre + write, hidden_delta)
                batch = summarize(native.flatten(0,1), finite.flatten(0,1), predicted_delta.flatten(0,1),
                                  hidden_delta.flatten(0,1))
                for name, value in batch.items(): collected[name].append(value.cpu())
            values = {name: torch.cat(parts).numpy() for name, parts in collected.items()}
            order = np.argsort(values["predicted_kl"])[::-1]; cutoff = max(1, len(order)//10)
            top_share = float(values["actual_kl"][order[:cutoff]].sum() / values["actual_kl"].sum())
            eligible_flip = values["actual_flip"].any() and (~values["actual_flip"]).any()
            flip_auc = fast_auc(values["actual_flip"], values["flip_score"]) if eligible_flip else None
            positive = values["predicted_kl"] > 1e-12
            ratio = values["actual_kl"][positive] / values["predicted_kl"][positive]
            results[corpus] = {
                "sites": len(order), "median_fisher_cosine": float(np.median(values["cosine"])),
                "fisher_kl_spearman": spearman(values["predicted_kl"], values["actual_kl"]),
                "hidden_norm_kl_spearman": spearman(values["hidden_norm"], values["actual_kl"]),
                "top_decile_kl_share": top_share, "flip_rate": float(values["actual_flip"].mean()),
                "flip_auc": float(flip_auc) if flip_auc is not None else None,
                "median_actual_over_predicted_kl": float(np.median(ratio)),
                "mean_actual_kl": float(values["actual_kl"].mean()),
                "mean_predicted_kl": float(values["predicted_kl"].mean()),
            }
    b0 = all(row["median_fisher_cosine"] >= .90 for row in results.values())
    b1 = all(row["fisher_kl_spearman"] >= .80 for row in results.values())
    b2 = all(row["top_decile_kl_share"] >= .50 for row in results.values())
    b3 = all(row["flip_auc"] is not None and row["flip_auc"] >= .85 for row in results.values())
    b4 = all(row["fisher_kl_spearman"] >= row["hidden_norm_kl_spearman"] + .20 for row in results.values())
    b5 = all(.5 <= row["median_actual_over_predicted_kl"] <= 2 for row in results.values())
    hypotheses = {
        "B0_local_jacobian_predicts_logit_direction": {"supported": b0},
        "B1_fisher_energy_ranks_finite_kl": {"supported": b1},
        "B2_top_decile_localizes_error": {"supported": b2},
        "B3_predicted_margin_identifies_top1_flips": {"supported": b3},
        "B4_output_geometry_beats_hidden_norm": {"supported": b4},
        "B5_second_order_scale_is_calibrated": {"supported": b5},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2, allow_nan=False))
    lines = ["# Blind sitewise prediction of sparse-graph errors", "",
             "For each natural site, the graph's omitted pooled hidden vector is propagated only through the analytic local Jacobian of final RMSNorm, unembedding, and bounded logits. The resulting Fisher energy and margin score are frozen predictions of the subsequently measured finite graph-replacement error.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Fisher cosine | Fisher/hidden Spearman with KL | Top-decile KL share | Flip rate / AUROC | Actual/predicted KL median |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['median_fisher_cosine']:.3f} | {row['fisher_kl_spearman']:.3f}/"
                     f"{row['hidden_norm_kl_spearman']:.3f} | {row['top_decile_kl_share']:.3f} | "
                     f"{row['flip_rate']:.3f}/{row['flip_auc']:.3f} | {row['median_actual_over_predicted_kl']:.3f} |")
    lines += ["", "This is a prospective calculation from the native state, omitted graph vector, and output weights; it does not use the finite replacement to choose sites or fit a scale. Success would make the graph diagnostically useful even where it is not exact. Failure would show that aggregate reconstruction hides nonlinear, example-specific errors requiring a richer downstream model."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
