#!/usr/bin/env python3
"""Fit fixed Layer-11 output subspaces directly for exact distributional KL."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_layer11_bottleneck_tail_replication import behavior_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_behavior_subspace.json")
SOURCE = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.npz"
OUTPUT = TENSOR_ROOT / "bilin12_layer11_behavior_subspace.json"
ARRAYS = TENSOR_ROOT / "bilin12_layer11_behavior_subspace.npz"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_BEHAVIOR_SUBSPACE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_behavior_subspace.png"


def make_basis(parameter: torch.Tensor, anchor: torch.Tensor | None) -> torch.Tensor:
    if anchor is None:
        return torch.linalg.qr(parameter, mode="reduced").Q
    residual = parameter - anchor @ (anchor.T @ parameter)
    tail = torch.linalg.qr(residual, mode="reduced").Q
    return torch.cat((anchor, tail), dim=1)


def canonical_overlap(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    values = torch.linalg.svdvals(first.T @ second)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "values": values.detach().cpu().tolist(),
    }


def load_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


@torch.inference_mode()
def collect_final_sites(model, layer: int, spec: dict, batch_size: int):
    assert layer == len(model.blocks) - 1
    rows = load_rows(spec)
    dimension = model.config.n_embd
    pres, writes, logits = [], [], []
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start : start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x = x0
        value_bus = None
        for block in model.blocks[:layer]:
            x, value_bus = block(x, value_bus, x0)
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = (mixed + attention)[:, -1].float()
        normalized = F.rms_norm(mixed + attention, (dimension,))
        write = (
            block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized))
            + block.mlp.Down_bias
        )[:, -1].float()
        raw_logits = model.lm_head(F.rms_norm(pre + write, (dimension,))).float()
        pres.append(pre)
        writes.append(write)
        logits.append(30 * torch.tanh(raw_logits / 30))
    native_logits = torch.cat(logits)
    native_logprob = F.log_softmax(native_logits, -1)
    return (
        torch.cat(pres),
        torch.cat(writes),
        native_logprob.exp(),
        native_logprob,
    )


def exact_kl_objective(
    model,
    pre: torch.Tensor,
    write: torch.Tensor,
    reference_probability: torch.Tensor,
    reference_logprob: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    dimension = write.shape[1]
    centered = write - mean
    projected = mean + (centered @ basis) @ basis.T
    raw = model.lm_head(F.rms_norm(pre + projected, (dimension,))).float()
    changed_logprob = F.log_softmax(30 * torch.tanh(raw / 30), -1)
    return (reference_probability * (reference_logprob - changed_logprob)).sum(-1).mean()


def validation_loss(model, data, mean, basis, batch_size: int) -> float:
    pre, write, probability, logprob = data
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(pre), batch_size):
            stop = start + batch_size
            loss = exact_kl_objective(
                model, pre[start:stop], write[start:stop], probability[start:stop],
                logprob[start:stop], mean, basis,
            )
            amount = len(pre[start:stop])
            total += float(loss) * amount
            count += amount
    return total / count


def optimize_basis(
    model,
    data,
    mean: torch.Tensor,
    initialization: torch.Tensor,
    anchor: torch.Tensor | None,
    config: dict,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameter = torch.nn.Parameter(
        initialization + config.get("initialization_noise", 1e-2) * torch.randn(
            initialization.shape, generator=generator, device="cuda"
        )
    )
    optimizer = torch.optim.Adam([parameter], lr=config["learning_rate"])
    pre, write, probability, logprob = data
    train_indices = torch.arange(0, len(pre), 2, device="cuda")
    validation_indices = torch.arange(1, len(pre), 2, device="cuda")
    validation_data = tuple(value[validation_indices] for value in data)
    # The exact unperturbed PCA initialization is a legitimate checkpoint.
    # Without this comparison an unstable optimizer can manufacture an
    # apparent negative result by returning a space worse than its baseline.
    with torch.no_grad():
        initial_basis = make_basis(initialization, anchor)
        best_validation = validation_loss(
            model, validation_data, mean, initial_basis,
            config["optimization_batch_size"],
        )
        best_basis = initial_basis.detach().clone()
    history = [{
        "step": -1,
        "train_batch_kl": None,
        "validation_kl": best_validation,
    }]
    with torch.enable_grad():
        for step in range(config["optimization_steps"]):
            sampled = train_indices[torch.randint(
                len(train_indices),
                (config["optimization_batch_size"],),
                generator=generator,
                device="cuda",
            )]
            basis = make_basis(parameter, anchor)
            loss = exact_kl_objective(
                model, pre[sampled], write[sampled], probability[sampled],
                logprob[sampled], mean, basis,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % config["validation_interval"] == 0 or step == config["optimization_steps"] - 1:
                with torch.no_grad():
                    current_basis = make_basis(parameter, anchor)
                    current_validation = validation_loss(
                        model, validation_data, mean, current_basis,
                        config["optimization_batch_size"],
                    )
                    if current_validation < best_validation:
                        best_validation = current_validation
                        best_basis = current_basis.detach().clone()
                history.append({
                    "step": step,
                    "train_batch_kl": float(loss.detach()),
                    "validation_kl": current_validation,
                })
    assert best_basis is not None
    return best_basis, {
        "seed": seed,
        "best_validation_kl": best_validation,
        "history": history,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    natural = torch.from_numpy(source["natural_basis"]).cuda().float()
    post_bias_mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for parameter in model.embedding.parameters():
        parameter.requires_grad_(False)
    for parameter in model.blocks.parameters():
        parameter.requires_grad_(False)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(False)

    print("collecting exact-KL discovery states", flush=True)
    discovery = collect_final_sites(
        model, config["layer"], config["discovery"], config["batch_size"]
    )
    fitted = {}
    metadata = {}
    for rank in config["ranks"]:
        seeds = (0, 1) if rank == config["replicated_rank"] else (0,)
        for family in ("global", "anchored"):
            for seed_offset in seeds:
                name = f"{family}{rank}_seed{seed_offset}"
                print("optimizing", name, flush=True)
                if family == "global":
                    initialization = natural[:, :rank]
                    anchor = None
                else:
                    initialization = tail[:, : rank - raw4.shape[1]]
                    anchor = raw4
                basis, row = optimize_basis(
                    model, discovery, post_bias_mean, initialization, anchor,
                    config, config["seed"] + 1000 * rank + 100 * (family == "anchored") + seed_offset,
                )
                fitted[name] = basis.cpu().numpy()
                metadata[name] = row

    bases = {
        "natural20": natural[:, :20].cpu().numpy(),
        "raw4_tail16": torch.cat((raw4, tail[:, :16]), 1).cpu().numpy(),
        "natural68": natural[:, :68].cpu().numpy(),
        "raw4_tail64": torch.cat((raw4, tail[:, :64]), 1).cpu().numpy(),
        **fitted,
    }
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("held-out evaluation", corpus, flush=True)
        behavior = behavior_metrics(
            model, config["layer"], spec, config["batch_size"],
            post_bias_mean.cpu().numpy(), bases,
        )
        _, write = collect_states_writes(
            model, config["layer"], spec, [-1], config["batch_size"]
        )
        centered = write.cpu().numpy() - source["pre_bias_write_mean"]
        variance = {
            name: float(np.square(centered @ basis).sum() / np.square(centered).sum())
            for name, basis in bases.items()
        }
        results[corpus] = {"behavior": behavior, "variance_fraction": variance}

    overlaps = {
        family: canonical_overlap(
            torch.from_numpy(fitted[f"{family}20_seed0"]).cuda(),
            torch.from_numpy(fitted[f"{family}20_seed1"]).cuda(),
        )
        for family in ("global", "anchored")
    }
    hypotheses = {
        "E1_anchored20_improves_tail_pca": {
            "supported": all(
                row["behavior"]["anchored20_seed0"]["median_kl"]
                <= 0.8 * row["behavior"]["raw4_tail16"]["median_kl"]
                for row in results.values()
            ),
        },
        "E2_global20_improves_natural_pca": {
            "supported": all(
                row["behavior"]["global20_seed0"]["median_kl"]
                <= 0.8 * row["behavior"]["natural20"]["median_kl"]
                for row in results.values()
            ),
        },
        "E3_rank68_reaches_near_sufficiency": {
            "supported": any(
                all(results[corpus]["behavior"][f"{family}68_seed0"]["median_kl"] <= 0.02 for corpus in results)
                for family in ("global", "anchored")
            ),
        },
        "E4_rank20_solutions_are_seed_stable": {
            "supported": all(row["mean"] >= 0.80 for row in overlaps.values()),
            "overlaps": overlaps,
        },
        "E5_improvement_replicates_across_seeds": {
            "supported": all(
                results[corpus]["behavior"][f"{family}20_seed{seed}"]["median_kl"]
                < results[corpus]["behavior"]["natural20" if family == "global" else "raw4_tail16"]["median_kl"]
                for corpus in results for family in ("global", "anchored") for seed in (0, 1)
            ),
        },
        "E6_weight_anchor_is_not_costly": {
            "supported": all(
                row["behavior"]["anchored20_seed0"]["median_kl"]
                <= 1.1 * row["behavior"]["global20_seed0"]["median_kl"]
                for row in results.values()
            ),
        },
        "E7_behavioral_fit_is_not_just_more_variance": {
            "supported": any(
                row["variance_fraction"]["global20_seed0"] < row["variance_fraction"]["natural20"]
                and row["behavior"]["global20_seed0"]["median_kl"] < row["behavior"]["natural20"]["median_kl"]
                for row in results.values()
            ),
        },
    }
    arrays = {name: basis.astype(np.float32) for name, basis in fitted.items()}
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "fits": metadata,
        "seed_overlaps": overlaps,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Layer-11 exact-behavior subspace red team",
        "",
        "Fixed output subspaces are optimized on 256 FineWeb training examples for exact native-to-reconstruction KL, selected on 256 interleaved validation examples, and frozen before evaluation on disjoint FineWeb, WikiText, and TinyStories. `global` fits may rotate all directions; `anchored` fits must retain the four weight-only spectral outliers and optimize only the remaining directions.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Behavior subspaces](bilin12_layer11_behavior_subspace.png)",
        "",
        "| Corpus | Natural-20 | Global-20 s0 / s1 | Raw4+tail16 | Anchored-20 s0 / s1 | Global-68 | Anchored-68 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        b = row["behavior"]
        lines.append(
            f"| {corpus} | {b['natural20']['median_kl']:.5f} | {b['global20_seed0']['median_kl']:.5f} / {b['global20_seed1']['median_kl']:.5f} | "
            f"{b['raw4_tail16']['median_kl']:.5f} | {b['anchored20_seed0']['median_kl']:.5f} / {b['anchored20_seed1']['median_kl']:.5f} | "
            f"{b['global68_seed0']['median_kl']:.5f} | {b['anchored68_seed0']['median_kl']:.5f} |"
        )
    lines += [
        "",
        f"Rank-20 seed overlap (mean canonical cosine): global **{overlaps['global']['mean']:.3f}**, anchored **{overlaps['anchored']['mean']:.3f}**.",
        "",
        "A successful fixed behavior subspace would show that the missing tail is compact but hidden by a variance metric. Failure would instead point toward context-routed subspaces, nonlinear reconstruction, or genuinely broad final-layer output support. Exact-KL optimization is deliberately allowed to sacrifice write variance, so variance and behavior can be separated directly.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(results)
    names = ("natural20", "global20_seed0", "raw4_tail16", "anchored20_seed0", "global68_seed0", "anchored68_seed0")
    labels = ("PCA20", "global20", "raw+tail16", "anchored20", "global68", "anchored68")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, corpus in zip(axes, corpora):
        values = [results[corpus]["behavior"][name]["median_kl"] for name in names]
        axis.bar(np.arange(len(names)), values)
        axis.axhline(0.02, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(names)), labels, rotation=35, ha="right")
        axis.set_yscale("log")
        axis.set_title(corpus)
        axis.set_ylabel("held-out median KL")
        axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
