#!/usr/bin/env python3
"""Finite suffix validation of weight-contracted scalar input decoders."""
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_residual_input_forms import collect, contract, quadratic, sorted_eigh
from redteam_bilin18_residual_input_linear import predict
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_taskblind_mlp_subspace_finite import intervene_write
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_residual_input_forms_finite"


def decoder(train, q, rank, floor_fraction=1e-8):
    mean = train.mean(0); centered = train - mean
    cov = centered.T @ centered / len(centered)
    ev, eu = torch.linalg.eigh(cov); ev = ev.clamp_min(ev.max() * floor_fraction)
    square = (eu * ev.sqrt()) @ eu.T; inverse = (eu * ev.rsqrt()) @ eu.T
    values, vectors = sorted_eigh(square @ q @ square)
    reads = inverse @ vectors[:, :rank]
    quadratic_mean = ((centered @ reads).square() * values[:rank]).sum(-1).mean()
    return {"mean": mean, "linear": 2 * q @ mean, "intercept": quadratic(train, q).mean() - quadratic_mean,
            "reads": reads, "values": values[:rank]}


def decode(x, fitted):
    centered = x - fitted["mean"]
    return fitted["intercept"] + centered @ fitted["linear"] + ((centered @ fitted["reads"]).square() * fitted["values"]).sum(-1)


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    config = json.loads((ROOT / "configs/bilin18_residual_input_forms_finite.json").read_text())
    source = np.load(ROOT / config["source"])
    affine_source = np.load(ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_residual_input_linear_redteam.npz")
    corpus = np.load(ROOT / config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float(); model.blocks.float(); model.lm_head.float()
    panel = collect(model, corpus, config["validation_start"], config)
    prepared = {}; saved = {}; coefficient_scores = {}
    rng = np.random.default_rng(config["seed"])
    for layer in config["layers"]:
        train = torch.from_numpy(source[f"discovery_inputs_L{layer}"]).double()
        directions = torch.from_numpy(source[f"directions_L{layer}"]).double()
        native = panel[layer]["writes"]; inputs = panel[layer]["inputs"]
        block = model.blocks[layer]
        left, right, down = [m.weight.detach().cpu().double() for m in (block.mlp.Left, block.mlp.Right, block.mlp.Down)]
        predictions = {"linear": [], **{f"quadratic{rank}": [] for rank in config["ranks"]}}
        for axis, direction in enumerate(directions):
            q = contract(left, right, down, direction)
            for rank in [0] + config["ranks"]:
                fitted = decoder(train, q, rank)
                name = "linear" if rank == 0 else f"quadratic{rank}"
                predictions[name].append(decode(inputs, fitted))
                for key, value in fitted.items(): saved[f"L{layer}_A{axis}_R{rank}_{key}"] = value.numpy()
        coefficients = native @ directions.T
        means = torch.from_numpy(source[f"discovery_writes_L{layer}"]).double().mean(0) @ directions.T
        remainder = native - coefficients @ directions
        predictions = {name: torch.stack(values, -1) for name, values in predictions.items()}
        affine = {key: torch.from_numpy(affine_source[f"L{layer}_{key}"]) for key in ("mean", "scale", "ym", "coef")}
        predictions["fitted_affine"] = predict(inputs, affine)
        predictions["permuted64"] = predictions["quadratic64"][rng.permutation(len(inputs))]
        conditions = {"native": native, "deleted": torch.zeros_like(native), "span_deleted": remainder,
                      "span_mean": remainder + means @ directions, "span_only": coefficients @ directions}
        conditions.update({name: remainder + coeff @ directions for name, coeff in predictions.items()})
        prepared[layer] = {name: value.cuda().float() for name, value in conditions.items()}
        coefficient_scores[str(layer)] = {name: float(1 - (coefficients - pred).square().sum() / (coefficients - means).square().sum()) for name, pred in predictions.items()}
        print(f"prepared L{layer}: {coefficient_scores[str(layer)]}", flush=True)
    methods = list(prepared[config["layers"][0]])
    metrics = {layer: {base: {name: empty_metric() for name in methods} for base in ("deleted", "span_deleted", "span_mean")} for layer in config["layers"]}
    maximum_error = 0.0
    for pindex, position in enumerate(config["positions"]):
        for offset in range(0, config["rows_per_panel"], config["batch_size"]):
            raw = np.array(corpus[config["validation_start"] + offset:config["validation_start"] + offset + config["batch_size"], :position + 2], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :-1], device="cuda"); targets = torch.as_tensor(raw[:, -1], device="cuda")
            reference = full_logits(model, ids)
            indices = torch.as_tensor(np.arange(offset, offset + len(raw)) * len(config["positions"]) + pindex, device="cuda")
            for layer in config["layers"]:
                conditions = {name: value[indices] for name, value in prepared[layer].items()}
                names, logits = intervene_write(model, ids, layer, conditions)
                out = dict(zip(names, logits))
                maximum_error = max(maximum_error, float((out["native"] - reference).abs().max()))
                for base in metrics[layer]:
                    for name in methods: update_metric(metrics[layer][base][name], out["native"], out[base], out[name], targets)
        print(f"finite position {position} complete", flush=True)
    results = {str(layer): {base: {name: finish(value) for name, value in conditions.items()} for base, conditions in rows.items()} for layer, rows in metrics.items()}
    gates = {
        "replay": maximum_error <= .02,
        "finite_dynamic": sum(results[str(l)]["span_mean"]["quadratic64"]["causal_fidelity"] >= .8 for l in config["layers"]) >= 3,
        "nonlinear_value": sum(results[str(l)]["span_mean"]["quadratic16"]["causal_fidelity"] - results[str(l)]["span_mean"]["linear"]["causal_fidelity"] >= .1 for l in (1, 8, 12)) >= 2,
        "binding": all(results[str(l)]["span_mean"]["quadratic64"]["causal_fidelity"] - results[str(l)]["span_mean"]["permuted64"]["causal_fidelity"] >= .1 for l in config["layers"]),
        "beats_fitted_affine": sum(results[str(l)]["span_mean"]["quadratic64"]["causal_fidelity"] - results[str(l)]["span_mean"]["fitted_affine"]["causal_fidelity"] >= .05 for l in (1,8,12)) >= 2
    }
    STEM.with_suffix(".json").write_text(json.dumps({"config": config, "gates": gates, "maximum_replay_error": maximum_error, "coefficient_r2": coefficient_scores, "results": results}, indent=2, allow_nan=False))
    np.savez_compressed(STEM.with_suffix(".npz"), **saved)
    lines = ["# Finite test of residual-direction input decoders", "", config["scope"], "", f"Locked gates: `{gates}`; maximum replay error `{maximum_error:.3g}`.", "", "Primary dynamic fidelity compares native logits with the mean-coefficient intervention. The native orthogonal remainder is retained; these numbers describe the four-axis component, not reconstruction of the full MLP.", "", "| Layer | Coefficient R2 rank64 | Dynamic logit linear/16/64 | Dynamic CE rank64 | Permuted64 dynamic logit | Four-axis-only full-MLP logit fidelity |", "|---|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; dynamic = row["span_mean"]
        values = "/".join(f"{dynamic[m]['causal_fidelity']:.3f}" for m in ("linear", "quadratic16", "quadratic64"))
        lines.append(f"| {layer} | {coefficient_scores[str(layer)]['quadratic64']:.3f} | {values} | {dynamic['quadratic64']['ce_effect_fidelity']:.3f} | {dynamic['permuted64']['causal_fidelity']:.3f} | {row['deleted']['span_only']['causal_fidelity']:.3f} |")
    lines += ["", "## Fitted affine alternative", "", "This control was added after the initial finite run. Ridge regularization was selected on row-grouped discovery validation; no finite-evaluation rows entered fitting.", "", "| Layer | Affine coefficient R2 | Affine dynamic logit fidelity | Rank64 advantage |", "|---|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]["span_mean"]
        lines.append(f"| {layer} | {coefficient_scores[str(layer)]['fitted_affine']:.3f} | {row['fitted_affine']['causal_fidelity']:.3f} | {row['quadratic64']['causal_fidelity'] - row['fitted_affine']['causal_fidelity']:+.3f} |")
    lines += ["", "The decoder reads normalized pre-MLP state; explaining how upstream layers generate those read coordinates remains open. Each scalar uses up to rank+1 dense input covectors, and four separate decoders are not a shared rank-64 module. The single row permutation is a binding diagnostic, not a permutation-tail significance estimate."]
    STEM.with_name("BILIN18_RESIDUAL_INPUT_FORMS_FINITE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gates), flush=True)


if __name__ == "__main__": main()
