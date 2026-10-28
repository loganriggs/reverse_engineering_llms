#!/usr/bin/env python3
"""Find spectral and state-predictable coordinates for dense late MLP operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_late_mlp_operator_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_late_mlp_operator_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_LATE_MLP_OPERATOR_ATLAS.md"


@torch.inference_mode()
def derivative_targets(block, states, axes, batch_size=16):
    """Return the complete local derivative [site, input_axis, output]."""
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    left_direction = axes.T.cuda() @ left_weight.T
    right_direction = axes.T.cuda() @ right_weight.T
    rows = []
    for start in range(0, len(states), batch_size):
        x = states[start:start + batch_size].cuda()
        left = x @ left_weight.T
        right = x @ right_weight.T
        gate = (right[:, :, None] * left_direction.T[None]
                + left[:, :, None] * right_direction.T[None])
        rows.append(torch.einsum("scn,dc->snd", gate, down).flatten(1).cpu())
    return torch.cat(rows)


def fidelity(target, prediction):
    return 1 - float((target.double() - prediction.double()).square().sum()
                     / target.double().square().sum().clamp_min(1e-30))


@torch.inference_mode()
def spectral_atlas(discovery, maximum_rank, null_replicates, generator):
    """Exact sample Gram spectrum and right singular modes through maximum_rank."""
    data = discovery.cuda()
    gram = data @ data.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    direct_energy = data.double().square().sum()
    gram_energy = eigenvalues.double().sum()
    accounting_error = float((gram_energy - direct_energy).abs() / direct_energy.clamp_min(1e-30))
    keep = min(maximum_rank, int((eigenvalues > eigenvalues[0] * 1e-10).sum()))
    singular = eigenvalues[:keep].sqrt()
    modes = ((eigenvectors[:, :keep].T @ data) / singular[:, None].clamp_min(1e-20)).cpu()
    shares = (eigenvalues.double() / direct_energy).cpu()
    probabilities = shares[shares > 0]
    effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    rank90 = int(torch.searchsorted(shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype)).item() + 1)
    null_top_shares = []
    for _ in range(null_replicates):
        # Independent signs preserve every coordinate magnitude while destroying
        # coherent cross-coordinate covariance between examples.
        signs = torch.randint(0, 2, data.shape, generator=generator, device="cpu", dtype=torch.int8)
        signed = data * signs.cuda().mul_(2).sub_(1)
        null_top = torch.linalg.eigvalsh(signed @ signed.T)[-1].double()
        null_top_shares.append(float(null_top / direct_energy))
    del data, gram
    return modes, {
        "energy_accounting_relative_error": accounting_error,
        "top_energy_share": float(shares[0]),
        "effective_rank": effective_rank,
        "discovery_rank90": rank90,
        "leading_singular_values": singular[:64].cpu().tolist(),
        "leading_energy_shares": shares[:64].tolist(),
        "sign_null_top_energy_shares": null_top_shares,
    }


@torch.inference_mode()
def project(target, modes):
    device_modes = modes.cuda()
    prediction = (target.cuda() @ device_modes.T) @ device_modes
    return prediction.cpu()


@torch.inference_mode()
def ridge_coefficients(train_states, train_coefficients, validation_states,
                       validation_target, modes, relative_alphas):
    x_train = train_states.float().cuda()
    z_train = train_coefficients.float().cuda()
    x_validation = validation_states.float().cuda()
    kernel = x_train @ x_train.T
    scale = float(torch.diagonal(kernel).mean().clamp_min(1e-30))
    cross = x_validation @ x_train.T
    identity = torch.eye(len(x_train), device="cuda")
    candidates = []
    for relative in relative_alphas:
        dual = torch.linalg.solve(kernel + relative * scale * identity, z_train)
        coefficients = cross @ dual
        prediction = coefficients @ modes.cuda()
        candidates.append((fidelity(validation_target, prediction.cpu()), relative, dual.cpu()))
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1], candidates[0][2], [
        {"relative_alpha": relative, "validation_fidelity": value}
        for value, relative, _ in sorted(candidates, key=lambda row: relative_alphas.index(row[1]))
    ]


@torch.inference_mode()
def predict(states, train_states, dual, modes):
    coefficients = (states.float().cuda() @ train_states.float().cuda().T) @ dual.cuda()
    return (coefficients @ modes.cuda()).cpu()


@torch.inference_mode(False)
def main():
    raw = json.loads(CONFIG.read_text())
    base = json.loads(Path(raw["source_config"]).read_text())
    config = {**base, **{key: value for key, value in raw.items()
                        if key not in ("source_config", "hypotheses")},
              "hypotheses": raw["hypotheses"]}
    native = json.loads(Path(config["native_reference"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])
    discovery_states = collect_states(model, config["atlas_discovery"], config)
    validation_states = collect_states(model, config["ridge_validation"], config)
    evaluation_states = {name: collect_states(model, spec, config)
                         for name, spec in config["evaluation"].items()}
    generator = torch.Generator().manual_seed(config["seed"])
    maximum_rank = max(config["spectral_budgets"])
    results = {}
    for layer in config["layers"]:
        print("atlas layer", layer, flush=True)
        discovery_target = derivative_targets(
            model.blocks[layer], discovery_states[layer], axes[layer], config["batch_size"])
        validation_target = derivative_targets(
            model.blocks[layer], validation_states[layer], axes[layer], config["batch_size"])
        modes, spectrum = spectral_atlas(
            discovery_target, maximum_rank, config["null_replicates"], generator)
        train_coefficients = discovery_target.cuda() @ modes.cuda().T
        alpha, dual, validation_curve = ridge_coefficients(
            discovery_states[layer], train_coefficients, validation_states[layer],
            validation_target, modes, config["ridge_relative_alphas"])
        corpus_rows = {}
        for corpus, states_by_layer in evaluation_states.items():
            target = derivative_targets(
                model.blocks[layer], states_by_layer[layer], axes[layer], config["batch_size"])
            budgets = {}
            for budget in config["spectral_budgets"]:
                local_modes = modes[:budget]
                oracle = fidelity(target, project(target, local_modes))
                predicted = fidelity(target, predict(
                    states_by_layer[layer], discovery_states[layer], dual[:, :budget], local_modes))
                budgets[str(budget)] = {
                    "oracle_projection_fidelity": oracle,
                    "state_prediction_fidelity": predicted,
                }
            corpus_rows[corpus] = {"sites": len(target), "budgets": budgets}
        results[str(layer)] = {
            "spectrum": spectrum,
            "selected_relative_alpha": alpha,
            "ridge_validation_curve_rank256": validation_curve,
            "corpora": corpus_rows,
        }
        del discovery_target, validation_target, modes, train_coefficients, dual
        torch.cuda.empty_cache()

    cells = [(layer, corpus, results[str(layer)]["corpora"][corpus])
             for layer in config["layers"] for corpus in config["evaluation"]]
    hypotheses = {
        "P0_train_spectral_energy_is_accounted": all(
            results[str(layer)]["spectrum"]["energy_accounting_relative_error"] <= 1e-5
            for layer in config["layers"]),
        "P1_rank64_operator_atlas_is_sufficient": all(
            row["budgets"]["64"]["oracle_projection_fidelity"] >= .80 for _, _, row in cells),
        "P2_rank128_operator_atlas_is_high_fidelity": all(
            row["budgets"]["128"]["oracle_projection_fidelity"] >= .90 for _, _, row in cells),
        "P3_rank128_state_predictor_is_sufficient": all(
            row["budgets"]["128"]["state_prediction_fidelity"] >= .80 for _, _, row in cells),
        "P4_state_prediction_nearly_matches_oracle_coordinates": all(
            row["budgets"]["128"]["state_prediction_fidelity"]
            >= row["budgets"]["128"]["oracle_projection_fidelity"] - .10
            for _, _, row in cells),
        "P5_state_predictor_transfers_ood": all(
            results[str(layer)]["corpora"][corpus]["budgets"]["128"]["state_prediction_fidelity"]
            >= results[str(layer)]["corpora"]["fineweb"]["budgets"]["128"]["state_prediction_fidelity"] - .10
            for layer in config["layers"] for corpus in ("wikitext", "tinystories")),
        "P6_spectral_rank64_beats_512_native_channels": all(
            row["budgets"]["64"]["oracle_projection_fidelity"]
            >= native["results"][corpus][str(layer)]["budgets"]["512"]["fidelity"] + .20
            for layer, corpus, row in cells),
        "P7_spectrum_has_nonrandom_outliers": all(
            results[str(layer)]["spectrum"]["top_energy_share"]
            >= 1.5 * max(results[str(layer)]["spectrum"]["sign_null_top_energy_shares"])
            for layer in config["layers"]),
    }
    native_advantage = (native["results"]["fineweb"]["16"]["budgets"]["512"]["fidelity"]
                        - max(native["results"]["fineweb"][str(layer)]["budgets"]["512"]["fidelity"]
                              for layer in (13, 14, 15)))
    spectral_advantage = (results["16"]["corpora"]["fineweb"]["budgets"]["64"]["oracle_projection_fidelity"]
                          - max(results[str(layer)]["corpora"]["fineweb"]["budgets"]["64"]["oracle_projection_fidelity"]
                                for layer in (13, 14, 15)))
    hypotheses["P8_layer16_is_more_native_than_spectral_special"] = (
        native_advantage >= .20 and spectral_advantage < .10)
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["P8_layer16_is_more_native_than_spectral_special"].update({
        "native_512_advantage": native_advantage, "spectral_rank64_advantage": spectral_advantage})
    payload = {"config": raw, "hypotheses": hypothesis_rows, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Spectral and state-predictable atlas of dense late-MLP operators", "",
             "The complete local bilinear derivative at each site is flattened as a 32×1,152 operator. Uncentered spectral modes are learned only on FineWeb. `Oracle` projects the already-known target into those modes and therefore measures coordinate compression, not prediction. `State` uses a validation-selected ridge map from the local normalized hidden state to the frozen spectral coordinates, then reconstructs the operator. This is the executable test.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Layer | Oracle R64 / R128 / R256 | State R64 / R128 / R256 | Native512 | Spectrum R90 | Top/null share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in config["evaluation"]:
        for layer in config["layers"]:
            row = results[str(layer)]["corpora"][corpus]["budgets"]
            oracle = " / ".join(f"{row[str(k)]['oracle_projection_fidelity']:.3f}" for k in (64, 128, 256))
            state = " / ".join(f"{row[str(k)]['state_prediction_fidelity']:.3f}" for k in (64, 128, 256))
            native512 = native["results"][corpus][str(layer)]["budgets"]["512"]["fidelity"]
            spectrum = results[str(layer)]["spectrum"]
            null = max(spectrum["sign_null_top_energy_shares"])
            lines.append(f"| {corpus} | {layer} | {oracle} | {state} | {native512:.3f} | {spectrum['discovery_rank90']} | {spectrum['top_energy_share']:.4f}/{null:.4f} |")
    lines += ["", "Oracle/state gaps are central: a small oracle atlas can merely rename each already-observed operator, whereas state prediction tests whether those coordinates form a reusable intermediate representation. Native-channel fidelity is copied from the separately frozen native-bank artifact and answers a different question: architectural execution sparsity."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
