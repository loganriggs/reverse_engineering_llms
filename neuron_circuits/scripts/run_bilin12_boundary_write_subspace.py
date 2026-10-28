#!/usr/bin/env python3
"""Test a compact newline-write subspace and state-conditioned coefficient router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_output_interface import effect_similarity, logits_from_hidden
from run_bilin12_boundary_state_contraction import paired_panel
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_boundary_output_interface import token_id
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from transformers import AutoTokenizer


CONFIG = Path("configs/bilin12_boundary_write_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_write_subspace.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_WRITE_SUBSPACE.md"
RAY = TENSOR_ROOT / "bilin12_boundary_state_contraction.json"


def features_delta(panel):
    pre, write, _, changed_pre, changed_write = panel
    dimension = pre.shape[-1]
    x = torch.nn.functional.rms_norm(changed_pre, (dimension,)) - torch.nn.functional.rms_norm(pre, (dimension,))
    return x.float().cpu(), (changed_write - write).float().cpu()


def fit_ridge(x, y, relative_penalty):
    x = x.double(); y = y.double()
    mean = x.mean(0); scale = x.std(0).clamp_min(1e-6)
    z = (x - mean) / scale; y_mean = y.mean(0); centered = y - y_mean
    gram = z @ z.T
    penalty = relative_penalty * float(torch.trace(gram)) / gram.shape[0]
    dual = torch.linalg.solve(
        gram + penalty * torch.eye(len(z), dtype=gram.dtype), centered,
    )
    weight = z.T @ dual
    return mean.float(), scale.float(), y_mean.float(), weight.float()


def predict(model, x):
    mean, scale, y_mean, weight = model
    return ((x - mean) / scale) @ weight + y_mean


@torch.inference_mode()
def causal_fidelity(model, panel, deltas):
    pre, write, base_logits, _, changed_write = panel
    oracle_logits = logits_from_hidden(model, pre + changed_write)
    predicted_logits = logits_from_hidden(model, pre + write + deltas.to(pre.device))
    return effect_similarity(base_logits, oracle_logits - base_logits,
                             predicted_logits - base_logits)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    state_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(state_config["parent_config"]).read_text())
    ray_result = json.loads(RAY.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = token_id(tokenizer, "\n")
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    train_panel = paired_panel(model, parent["semantic_discovery"], parent["layer"],
                               newline, parent["batch_size"])
    calibration_panel = paired_panel(model, parent["calibration"], parent["layer"],
                                     newline, parent["batch_size"])
    x_train, delta_train = features_delta(train_panel)
    x_cal, delta_cal = features_delta(calibration_panel)
    _, _, vh = torch.linalg.svd(delta_train, full_matrices=False)
    max_rank = max(config["ranks"]); basis_all = vh[:max_rank].T.contiguous()
    generator = torch.Generator().manual_seed(config["seed"])
    random_basis = torch.linalg.qr(torch.randn(delta_train.shape[1], max_rank,
                                                generator=generator)).Q
    primary = config["primary_rank"]
    basis = basis_all[:, :primary]
    target_train = delta_train @ basis
    target_cal = delta_cal @ basis
    candidates = []
    for penalty in config["ridge_relative_penalties"]:
        fitted = fit_ridge(x_train, target_train, penalty)
        mse = float((predict(fitted, x_cal) - target_cal).square().mean())
        candidates.append((mse, penalty, fitted))
    calibration_mse, penalty, router = min(candidates, key=lambda row: row[0])
    shuffled_target = target_train[torch.randperm(len(target_train), generator=generator)]
    shuffled_router = fit_ridge(x_train, shuffled_target, penalty)

    results = {}
    for corpus, spec in parent["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, parent["layer"], newline, parent["batch_size"])
        x, delta = features_delta(panel)
        frontier = {}
        for rank in config["ranks"]:
            learned = basis_all[:, :rank]
            random = random_basis[:, :rank]
            learned_delta = (delta @ learned) @ learned.T
            random_delta = (delta @ random) @ random.T
            learned_cosine, learned_fidelity = causal_fidelity(model, panel, learned_delta)
            random_cosine, random_fidelity = causal_fidelity(model, panel, random_delta)
            frontier[str(rank)] = {
                "learned_cosine": learned_cosine, "learned_fidelity": learned_fidelity,
                "random_cosine": random_cosine, "random_fidelity": random_fidelity,
                "write_energy_fraction_median": float(
                    (learned_delta.square().sum(-1)
                     / delta.square().sum(-1).clamp_min(1e-30)).median()),
            }
        oracle_delta = (delta @ basis) @ basis.T
        routed_delta = predict(router, x) @ basis.T
        shuffled_delta = predict(shuffled_router, x) @ basis.T
        oracle_cosine, oracle_fidelity = causal_fidelity(model, panel, oracle_delta)
        routed_cosine, routed_fidelity = causal_fidelity(model, panel, routed_delta)
        shuffled_cosine, shuffled_fidelity = causal_fidelity(model, panel, shuffled_delta)
        results[corpus] = {"examples": len(x), "frontier": frontier,
                           "oracle": {"cosine": oracle_cosine, "fidelity": oracle_fidelity},
                           "routed": {"cosine": routed_cosine, "fidelity": routed_fidelity},
                           "shuffled": {"cosine": shuffled_cosine, "fidelity": shuffled_fidelity}}
    s0 = all(row["oracle"]["fidelity"] >= .80 for row in results.values())
    s1 = any(all(results[corpus]["frontier"][str(rank)]["learned_fidelity"] >= .80
                 for corpus in results) for rank in config["ranks"] if rank <= 32)
    s2 = all(row["oracle"]["fidelity"]
             >= row["frontier"][str(primary)]["random_fidelity"] + .30
             for row in results.values())
    s3 = all(row["routed"]["fidelity"] >= .70
             and row["routed"]["fidelity"] >= row["oracle"]["fidelity"] - .15
             for row in results.values())
    s4 = all(row["routed"]["fidelity"] >= row["shuffled"]["fidelity"] + .20
             for row in results.values())
    previous_wiki = ray_result["results"]["wikitext"]["oracle"]["fidelity"]
    s5 = results["wikitext"]["oracle"]["fidelity"] >= previous_wiki + .25
    hypotheses = {
        "S0_rank16_oracle_subspace_is_sufficient": {"supported": s0},
        "S1_compact_frontier_reaches_ood": {"supported": s1},
        "S2_learned_subspace_beats_random": {"supported": s2},
        "S3_state_router_recovers_compact_module": {"supported": s3},
        "S4_router_beats_shuffled_coefficients": {"supported": s4},
        "S5_subspace_repairs_wikitext_direction_rotation": {"supported": s5,
                                                              "previous": previous_wiki},
    }
    OUTPUT.write_text(json.dumps({"config": config, "selected_penalty": penalty,
                                  "calibration_mse": calibration_mse,
                                  "hypotheses": hypotheses, "results": results}, indent=2))
    lines = ["# Compact boundary-write module and state router", "",
             "A fixed FineWeb SVD basis of exact Layer-11 newline write changes is evaluated with oracle coefficients and with coefficients predicted from upstream state changes. Equal-rank random bases and shuffled routers are controls.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Primary rank `{primary}`; selected ridge penalty `{penalty:g}`; calibration coefficient MSE `{calibration_mse:.4g}`.", "",
              "| Corpus | Rank-16 oracle/random fidelity | Routed/shuffled fidelity | Rank-16 energy | Smallest rank with fidelity >=0.80 |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        primary_row = row["frontier"][str(primary)]
        passing = [rank for rank in config["ranks"]
                   if row["frontier"][str(rank)]["learned_fidelity"] >= .80]
        lines.append(
            f"| {corpus} | {row['oracle']['fidelity']:.3f}/{primary_row['random_fidelity']:.3f} | "
            f"{row['routed']['fidelity']:.3f}/{row['shuffled']['fidelity']:.3f} | "
            f"{primary_row['write_energy_fraction_median']:.3f} | {min(passing) if passing else 'none'} |"
        )
    lines += ["", "Oracle projection asks whether a fixed compact dictionary contains the needed directions. The router asks the separate question of whether a simple upstream-state map selects the right combination."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
