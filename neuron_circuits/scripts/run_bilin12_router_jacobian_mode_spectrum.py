#!/usr/bin/env python3
"""Find a compact spectral dictionary of weight-derived transport Jacobians."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import analytic_position
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_jacobian_mode_spectrum.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_jacobian_mode_spectrum.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JACOBIAN_MODE_SPECTRUM.md"
FIGURE = TENSOR_ROOT / "bilin12_router_jacobian_mode_spectrum.png"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def collect(model, spec, config, axes):
    values = []; source = load_rows(spec)
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        for position in config["positions"]:
            values.append(analytic_position(model, ids, position, config, axes, False).cpu())
    return torch.cat(values).reshape(-1, 1024)


def reconstruction_fidelity(values, mean, modes, rank):
    centered = values - mean
    if rank == 0: prediction = mean.expand_as(values)
    else:
        basis = modes[:rank]
        prediction = mean + (centered @ basis.T) @ basis
    return 1 - float((values - prediction).square().sum() / values.square().sum().clamp_min(1e-30))


def shuffled_top(centered, generator):
    rows, columns = centered.shape
    keys = torch.rand(rows, columns, generator=generator, device=centered.device)
    order = keys.argsort(0)
    shuffled = centered.gather(0, order)
    # Only the leading value is needed; randomized PCA avoids repeated full SVDs.
    _, singular, _ = torch.pca_lowrank(shuffled, q=8, center=False, niter=4)
    return float(singular[0])


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (basis @ rotation.detach()).float()
    print("collect mode train", flush=True); train = collect(model, config["mode_train"], config, axes)
    mean = train.mean(0); centered = (train - mean).cuda()
    print("svd", flush=True); _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    variance = singular.square(); cumulative = variance.cumsum(0) / variance.sum()
    probabilities = variance / variance.sum(); effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()))
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    null_tops = [shuffled_top(centered, generator) for _ in range(config["null_replicates"])]
    modes = vh[:max(config["ranks"])].cpu(); results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); values = collect(model, spec, config, axes)
        results[corpus] = {"sites": len(values), "mean_only_fidelity": reconstruction_fidelity(values, mean, modes, 0),
                           "rank_fidelity": {str(rank): reconstruction_fidelity(values, mean, modes, rank)
                                             for rank in config["ranks"]}}
    p0 = float(cumulative[15]) >= .80
    p1 = all(row["rank_fidelity"]["32"] >= .95 for row in results.values())
    p2 = all(row["rank_fidelity"]["64"] >= .98 for row in results.values())
    fw = results["fineweb"]["rank_fidelity"]["32"]
    p3 = all(results[c]["rank_fidelity"]["32"] >= fw - .03 for c in ("wikitext", "tinystories"))
    p4 = float(singular[0]) >= 1.5 * max(null_tops)
    p5 = all(row["mean_only_fidelity"] < .90 for row in results.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "P0_jacobian_variation_is_low_rank": p0, "P1_rank32_dictionary_is_high_fidelity": p1,
        "P2_rank64_dictionary_is_nearly_complete": p2, "P3_mode_dictionary_transfers_ood": p3,
        "P4_spectrum_has_random_matrix_outliers": p4, "P5_mean_transport_is_not_enough": p5}.items()}
    spectrum = {"training_sites": len(train), "effective_rank": effective_rank,
                "centered_cumulative_variance": {str(rank): float(cumulative[rank - 1]) for rank in config["ranks"]},
                "leading_singular_values": singular[:32].tolist(), "null_leading_values": null_tops,
                "leading_to_max_null_ratio": float(singular[0]) / max(null_tops)}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "spectrum": spectrum,
                                  "results": results}, indent=2, allow_nan=False))
    fig, axes_plot = plt.subplots(1, 2, figsize=(11, 4.2))
    axes_plot[0].loglog(np.arange(1, 129), singular[:128].cpu(), marker=".", label="natural Jacobians")
    for value in null_tops: axes_plot[0].axhline(value, color="gray", alpha=.35)
    axes_plot[0].set(title="Centered Jacobian spectrum", xlabel="mode", ylabel="singular value"); axes_plot[0].legend()
    for corpus, row in results.items():
        axes_plot[1].plot(config["ranks"], [row["rank_fidelity"][str(rank)] for rank in config["ranks"]], marker="o", label=corpus)
    axes_plot[1].set(title="Frozen FineWeb mode dictionary", xlabel="modes", ylabel="total Jacobian fidelity", xscale="log"); axes_plot[1].legend(); axes_plot[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Spectrum of natural transport Jacobians", "",
             "Weight-derived 32-by-32 Layer-6-to-final MLP transport Jacobians were collected at 1,024 FineWeb sites. PCA was fit only there, then its mean and mode dictionary were frozen and tested on disjoint FineWeb and two OOD corpora. Column-wise shuffled matrices preserve every marginal entry distribution while destroying coordinated circuit modes.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Jacobian mode spectrum](bilin12_router_jacobian_mode_spectrum.png)", "",
              f"Centered effective rank: **{effective_rank:.1f}**. Leading/null ratio: **{spectrum['leading_to_max_null_ratio']:.2f}**. Rank-16 centered variance: **{float(cumulative[15]):.3f}**.", "",
              "| Corpus | Mean only | Rank 16 | Rank 32 | Rank 64 | Rank 128 |", "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['mean_only_fidelity']:.3f} | " + " | ".join(f"{row['rank_fidelity'][str(rank)]:.3f}" for rank in (16,32,64,128)) + " |")
    lines += ["", "This separates two problems: a compact mode dictionary may exist even when predicting its coefficients from local semantic variables remains difficult. Spectral outliers support reusable coordinated transport modes; they do not by themselves assign human meanings to those modes."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
