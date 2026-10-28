#!/usr/bin/env python3
"""Replicate full-tensor power-law and spike regimes on an independent model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw, spectrum_from_gram
from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_checkpoint_spectrum_replication.json")
OUTPUT = TENSOR_ROOT / "bilin12_checkpoint_spectrum_replication.json"
ARRAYS = TENSOR_ROOT / "bilin12_checkpoint_spectrum_replication.npz"
REPORT = TENSOR_ROOT / "BILIN12_CHECKPOINT_SPECTRUM_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin12_checkpoint_spectrum_replication.png"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    snapshot = Path(config["snapshot"])
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    model_config = json.loads((snapshot / "config.json").read_text())
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}; arrays = {}
    for layer in config["layers"]:
        print("independent spectrum layer", layer, flush=True)
        prefix = f"transformer.h.{layer}.mlp."
        left = state[prefix + "Left.weight"].cuda()
        right = state[prefix + "Right.weight"].cuda()
        down = state[prefix + "Down.weight"].cuda()
        channel_gram = symmetric_channel_gram(left, right)
        observed = spectrum_from_gram(down, channel_gram).cpu().numpy()
        arrays[f"layer{layer}_observed"] = observed
        sign_rows = []; pairing_rows = []
        for replicate in range(config["null_replicates"]):
            signs = torch.randint(0, 2, (down.shape[1],), generator=generator, device="cuda") * 2 - 1
            sign = spectrum_from_gram(down * signs, channel_gram).cpu().numpy()
            permutation = torch.randperm(down.shape[1], generator=generator, device="cuda")
            pairing = spectrum_from_gram(down[:, permutation], channel_gram).cpu().numpy()
            arrays[f"layer{layer}_sign{replicate}"] = sign
            arrays[f"layer{layer}_pairing{replicate}"] = pairing
            sign_rows.append(fit_powerlaw(sign, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"]))
            pairing_rows.append(fit_powerlaw(pairing, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"]))
        observed_fit = fit_powerlaw(observed, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"])
        results[str(layer)] = {
            "observed": observed_fit,
            "sign_nulls": sign_rows, "pairing_nulls": pairing_rows,
            "sign_summary": {key: float(np.median([row[key] for row in sign_rows])) for key in ("alpha", "r_squared", "mean_top_log_excess", "top4_energy_fraction")},
            "pairing_summary": {key: float(np.median([row[key] for row in pairing_rows])) for key in ("alpha", "r_squared", "mean_top_log_excess", "top4_energy_fraction")},
        }
        del left, right, down, channel_gram
    early, middle, late = map(str, config["layers"])
    hypotheses = {
        "P1_powerlaw_bulk_replicates": {"supported": all(row["observed"]["r_squared"] >= .9 for row in results.values())},
        "P2_nulls_retain_bulk": {"supported": all(row[family]["r_squared"] >= .9 and abs(row[family]["alpha"] - row["observed"]["alpha"]) <= .15 for row in results.values() for family in ("sign_summary", "pairing_summary"))},
        "P3_late_has_four_strict_spikes": {"supported": all(value > 0 for value in results[late]["observed"]["top_log_excess"]), "excess": results[late]["observed"]["top_log_excess"]},
        "P4_late_exceeds_coordination_nulls": {"supported": results[late]["observed"]["mean_top_log_excess"] > max(row["mean_top_log_excess"] for family in ("sign_nulls", "pairing_nulls") for row in results[late][family])},
        "P5_late_is_more_concentrated": {"supported": results[late]["observed"]["top4_energy_fraction"] - max(results[early]["observed"]["top4_energy_fraction"], results[middle]["observed"]["top4_energy_fraction"]) >= .005},
        "P6_middle_lacks_strict_spikes": {"supported": all(value <= 0 for value in results[middle]["observed"]["top_log_excess"]), "excess": results[middle]["observed"]["top_log_excess"]},
        "P7_depth_excess_pattern_replicates": {"supported": results[late]["observed"]["mean_top_log_excess"] - max(results[early]["observed"]["mean_top_log_excess"], results[middle]["observed"]["mean_top_log_excess"]) >= .1},
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({"config": config, "model_config": model_config, "results": results, "hypotheses": hypotheses}, indent=2, allow_nan=False))
    lines = [
        "# Independent-checkpoint replication of full tensor spectra", "",
        f"This repeats the frozen full-width tensor-map spectrum and coordination-null analysis on `{config['model_repo']}` at commit `{config['snapshot_commit']}` (12 layers, width 768, six heads, training step {model_config['step']}). It is an independently trained checkpoint, not another corpus or layer from the 18-layer model.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Independent checkpoint spectra](bilin12_checkpoint_spectrum_replication.png)", "", "| Layer | Alpha / R² | Top excesses | Mean top-4 | Top-4 energy | Sign / pairing null mean |", "|---:|---:|---:|---:|---:|---:|"]
    for layer, row in results.items():
        observed = row["observed"]
        lines.append(f"| {layer} | {observed['alpha']:.3f} / {observed['r_squared']:.3f} | {' / '.join(f'{value:.3f}' for value in observed['top_log_excess'])} | {observed['mean_top_log_excess']:.3f} | {observed['top4_energy_fraction']:.3f} | {row['sign_summary']['mean_top_log_excess']:.3f} / {row['pairing_summary']['mean_top_log_excess']:.3f} |")
    lines += ["", "The power-law fit and spike criterion are deliberately unchanged in form but rescaled in rank window for width 768. Replication of a late spike would establish an architectural/training-family regularity; it would still require natural-state and causal validation before being called a circuit."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, layer in zip(axes, config["layers"]):
        observed = arrays[f"layer{layer}_observed"]
        ranks = np.arange(1, len(observed) + 1)
        sign = np.stack([arrays[f"layer{layer}_sign{i}"] for i in range(config["null_replicates"])])
        pairing = np.stack([arrays[f"layer{layer}_pairing{i}"] for i in range(config["null_replicates"])])
        axis.loglog(ranks, observed, label="observed", linewidth=2)
        axis.loglog(ranks, np.median(sign, 0), label="sign null")
        axis.loglog(ranks, np.median(pairing, 0), label="pairing null")
        axis.axvspan(config["bulk_rank_start"], config["bulk_rank_stop"], color="grey", alpha=.1)
        axis.set_title(f"Layer {layer}"); axis.set_xlabel("spectral rank"); axis.set_ylabel("singular value"); axis.legend()
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
