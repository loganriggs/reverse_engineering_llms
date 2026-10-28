#!/usr/bin/env python3
"""Predict per-example contraction of a generic newline MLP-write ray."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_output_interface import (
    effect_similarity, final_components, logits_from_hidden, natural_prototype, token_id,
)
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_state_contraction.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_state_contraction.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_STATE_CONTRACTION.md"
FIXED = TENSOR_ROOT / "bilin12_boundary_output_interface.json"


@torch.inference_mode()
def paired_panel(model, spec, layer, newline, batch_size):
    rows = load_rows(spec)
    columns = [[] for _ in range(5)]
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        keep = ids[:, -1] != newline
        if not bool(keep.any()):
            continue
        ids = ids[keep]; changed = ids.clone(); changed[:, -1] = newline
        pre, write, logits = final_components(model, ids, layer)
        changed_pre, changed_write, _ = final_components(model, changed, layer)
        for store, value in zip(columns, (pre, write, logits, changed_pre, changed_write)):
            store.append(value)
    return tuple(torch.cat(values) for values in columns)


def features_targets(panel, ray):
    pre, write, _, changed_pre, changed_write = panel
    dimension = pre.shape[-1]
    state_delta = F.rms_norm(changed_pre, (dimension,)) - F.rms_norm(pre, (dimension,))
    write_delta = changed_write - write
    coefficient = (write_delta @ ray) / ray.square().sum().clamp_min(1e-30)
    residual = write_delta - coefficient[:, None] * ray
    energy_fraction = 1 - residual.square().sum(-1) / write_delta.square().sum(-1).clamp_min(1e-30)
    return state_delta.float().cpu(), coefficient.float().cpu(), energy_fraction.float().cpu()


def fit_ridge(x, y, relative_penalty):
    x = x.double(); y = y.double()
    mean = x.mean(0); scale = x.std(0).clamp_min(1e-6)
    z = (x - mean) / scale; y_mean = y.mean(); centered_y = y - y_mean
    gram = z @ z.T
    penalty = relative_penalty * float(torch.trace(gram)) / gram.shape[0]
    dual = torch.linalg.solve(
        gram + penalty * torch.eye(len(z), dtype=gram.dtype), centered_y,
    )
    weight = z.T @ dual
    return mean.float(), scale.float(), y_mean.float(), weight.float()


def predict(model, x):
    mean, scale, y_mean, weight = model
    return ((x - mean) / scale) @ weight + y_mean


def scalar_metrics(target, predicted):
    target = target.numpy(); predicted = predicted.numpy()
    correlation = float(np.corrcoef(target, predicted)[0, 1])
    denominator = np.square(target - target.mean()).sum()
    r2 = 1 - float(np.square(target - predicted).sum() / max(denominator, 1e-30))
    return {"pearson": correlation, "r2": r2}


@torch.inference_mode()
def causal_metrics(model, panel, ray, coefficients):
    pre, write, base_logits, _, changed_write = panel
    oracle_logits = logits_from_hidden(model, pre + changed_write)
    oracle_effect = oracle_logits - base_logits
    patched = logits_from_hidden(model, pre + write + coefficients.to(pre.device)[:, None] * ray)
    return effect_similarity(base_logits, oracle_effect, patched - base_logits)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    fixed = json.loads(FIXED.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = token_id(tokenizer, "\n")
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    ray, count = natural_prototype(
        model, parent["prototype"], parent["layer"], parent["positions"], newline,
        parent["batch_size"],
    )
    train_panel = paired_panel(
        model, parent["semantic_discovery"], parent["layer"], newline, parent["batch_size"],
    )
    calibration_panel = paired_panel(
        model, parent["calibration"], parent["layer"], newline, parent["batch_size"],
    )
    x_train, y_train, _ = features_targets(train_panel, ray)
    x_cal, y_cal, _ = features_targets(calibration_panel, ray)
    candidates = []
    for penalty in config["ridge_relative_penalties"]:
        fitted = fit_ridge(x_train, y_train, penalty)
        mse = float((predict(fitted, x_cal) - y_cal).square().mean())
        candidates.append((mse, penalty, fitted))
    calibration_mse, penalty, fitted = min(candidates, key=lambda row: row[0])
    generator = torch.Generator().manual_seed(config["seed"])
    shuffled_y = y_train[torch.randperm(len(y_train), generator=generator)]
    shuffled = fit_ridge(x_train, shuffled_y, penalty)

    results = {}
    fixed_scale = fixed["generic_calibration"]["scale"]
    for corpus, spec in parent["evaluation"].items():
        print("evaluate", corpus, flush=True)
        panel = paired_panel(model, spec, parent["layer"], newline, parent["batch_size"])
        x, y, energy = features_targets(panel, ray)
        predicted = predict(fitted, x)
        shuffled_predicted = predict(shuffled, x)
        oracle_cosine, oracle_fidelity = causal_metrics(model, panel, ray, y)
        predicted_cosine, predicted_fidelity = causal_metrics(model, panel, ray, predicted)
        shuffled_cosine, shuffled_fidelity = causal_metrics(model, panel, ray, shuffled_predicted)
        fixed_coefficients = torch.full_like(y, fixed_scale)
        fixed_cosine, fixed_fidelity = causal_metrics(model, panel, ray, fixed_coefficients)
        results[corpus] = {
            "examples": len(y), "coefficient": scalar_metrics(y, predicted),
            "shuffled_coefficient": scalar_metrics(y, shuffled_predicted),
            "write_energy_fraction_median": float(energy.median()),
            "oracle": {"cosine": oracle_cosine, "fidelity": oracle_fidelity},
            "predicted": {"cosine": predicted_cosine, "fidelity": predicted_fidelity},
            "shuffled": {"cosine": shuffled_cosine, "fidelity": shuffled_fidelity},
            "fixed": {"cosine": fixed_cosine, "fidelity": fixed_fidelity},
        }
    t0 = all(row["oracle"]["fidelity"] >= .70 for row in results.values())
    t1 = results["wikitext"]["oracle"]["fidelity"] >= results["wikitext"]["fixed"]["fidelity"] + .20
    t2 = all(row["coefficient"]["pearson"] >= .30 and row["coefficient"]["r2"] > 0
             for row in results.values())
    t3 = all(row["predicted"]["fidelity"] >= .60
             and row["predicted"]["fidelity"] >= row["oracle"]["fidelity"] - .15
             for row in results.values())
    t4 = results["wikitext"]["predicted"]["fidelity"] >= results["wikitext"]["fixed"]["fidelity"] + .15
    t5 = all(row["predicted"]["fidelity"] >= row["shuffled"]["fidelity"] + .20
             for row in results.values())
    hypotheses = {
        "T0_one_effect_ray_is_oracle_sufficient": {"supported": t0},
        "T1_wikitext_failure_is_mostly_scale": {"supported": t1},
        "T2_state_change_predicts_contraction": {"supported": t2},
        "T3_predicted_contraction_is_causally_sufficient": {"supported": t3},
        "T4_state_contraction_repairs_wikitext": {"supported": t4},
        "T5_real_state_mapping_beats_shuffled_control": {"supported": t5},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "ray_prototype_count": count,
        "selected_penalty": penalty, "calibration_mse": calibration_mse,
        "hypotheses": hypotheses, "results": results,
    }, indent=2))
    lines = ["# State contraction of the natural newline write ray", "",
             "A frozen Layer-11 newline write direction is given either its per-example oracle coefficient or a coefficient predicted linearly from the paired upstream state change. The predictor is trained on FineWeb and compared with a shuffled-target fit.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected relative ridge penalty: `{penalty:g}`; calibration scalar MSE: `{calibration_mse:.4g}`.", "",
              "| Corpus | Coefficient Pearson/R2 | Ray write-energy fraction | Fixed fidelity | Oracle-ray fidelity | Predicted fidelity | Shuffled fidelity |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {row['coefficient']['pearson']:.3f}/{row['coefficient']['r2']:.3f} | "
            f"{row['write_energy_fraction_median']:.3f} | {row['fixed']['fidelity']:.3f} | "
            f"{row['oracle']['fidelity']:.3f} | {row['predicted']['fidelity']:.3f} | "
            f"{row['shuffled']['fidelity']:.3f} |"
        )
    lines += ["", "Oracle scaling tests whether the direction is sufficient; predicted scaling tests whether a simple reusable state-contraction rule can recover it. These are deliberately separate claims."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
