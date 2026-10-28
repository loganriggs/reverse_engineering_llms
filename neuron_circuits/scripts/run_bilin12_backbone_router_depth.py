#!/usr/bin/env python3
"""Locate when a compact late-write program becomes linearly readable.

A single rank-32 write basis is learned from four FineWeb token edits.  For a
token-to-newline edit, frozen ridge maps predict the final MLP write
coefficients from local states at the embedding boundary and after Layers
0--10.  Delta-only, paired-state, coordinate-polynomial, and explicit-carrier
features are compared.  Causal fidelity is measured by inserting each
predicted write delta into the original final-layer computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_backbone_router_depth.json")
OUTPUT = TENSOR_ROOT / "bilin12_backbone_router_depth.json"
REPORT = TENSOR_ROOT / "BILIN12_BACKBONE_ROUTER_DEPTH.md"
FIGURE = TENSOR_ROOT / "bilin12_backbone_router_depth.png"
DEPTHS = tuple(range(-1, 11))


def fit_ridge(x, y, relative_penalty):
    """Dual ridge with train-only standardization and an explicit intercept."""
    x = x.double(); y = y.double()
    mean = x.mean(0); scale = x.std(0).clamp_min(1e-6)
    z = (x - mean) / scale
    y_mean = y.mean(0); centered = y - y_mean
    gram = z @ z.T
    penalty = relative_penalty * float(torch.trace(gram)) / len(z)
    dual = torch.linalg.solve(
        gram + penalty * torch.eye(len(z), dtype=gram.dtype), centered,
    )
    return mean.float(), scale.float(), y_mean.float(), (z.T @ dual).float()


def predict(fitted, x):
    mean, scale, y_mean, weight = fitted
    return ((x - mean) / scale) @ weight + y_mean


def coefficient_metrics(target, prediction):
    centered = target - target.mean(0)
    denominator = centered.square().sum().clamp_min(1e-30)
    r2 = 1 - (target - prediction).square().sum() / denominator
    cosine = F.cosine_similarity(target, prediction, dim=-1)
    return {"r2": float(r2), "median_cosine": float(cosine.median())}


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def spearman(values):
    return float(np.corrcoef(rankdata(DEPTHS), rankdata(values))[0, 1])


@torch.inference_mode()
def trace_batch(model, ids):
    """Return local carrier states at input/0--10 and final-layer components."""
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    rows = [{"residual": x[:, -1], "embedding": x0[:, -1],
             "value_bus": torch.zeros_like(x[:, -1])}]
    final_pre = final_write = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + write
        if layer <= 10:
            rows.append({"residual": x[:, -1], "embedding": x0[:, -1],
                         "value_bus": value_bus[:, -1].reshape(len(ids), dimension)})
        if layer == 11:
            final_pre, final_write = pre[:, -1], write[:, -1]
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    logits = 30 * torch.tanh(raw / 30)
    return rows, final_pre, final_write, logits


def make_features(base, changed):
    dimension = base["residual"].shape[-1]
    base_residual = F.rms_norm(base["residual"], (dimension,)).float().cpu()
    changed_residual = F.rms_norm(changed["residual"], (dimension,)).float().cpu()
    delta = changed_residual - base_residual
    embedding_delta = (changed["embedding"] - base["embedding"]).float().cpu()
    bus_delta = (changed["value_bus"] - base["value_bus"]).float().cpu()
    return {
        "delta": delta,
        "pair": torch.cat((base_residual, delta), -1),
        "polynomial": torch.cat((base_residual, delta, base_residual * delta,
                                  delta.square()), -1),
        "carriers": torch.cat((base_residual, delta, embedding_delta, bus_delta), -1),
    }


@torch.inference_mode()
def collect(model, spec, target, batch_size, sequence_length):
    rows = load_rows(spec)
    feature_rows = {depth: {} for depth in DEPTHS}
    stores = {depth: {} for depth in DEPTHS}
    final_columns = [[] for _ in range(5)]
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        keep = ids[:, -1] != target
        if not bool(keep.any()):
            continue
        ids = ids[keep]; changed_ids = ids.clone(); changed_ids[:, -1] = target
        joined = torch.cat((ids, changed_ids))
        traces, pre, write, logits = trace_batch(model, joined)
        count = len(ids)
        for depth, row in zip(DEPTHS, traces):
            base = {key: value[:count] for key, value in row.items()}
            changed = {key: value[count:] for key, value in row.items()}
            features = make_features(base, changed)
            for family, values in features.items():
                stores[depth].setdefault(family, []).append(values)
        for store, value in zip(final_columns, (pre[:count], write[:count], logits[:count],
                                                  pre[count:], write[count:])):
            store.append(value)
    for depth in DEPTHS:
        feature_rows[depth] = {family: torch.cat(values) for family, values in stores[depth].items()}
    panel = tuple(torch.cat(values) for values in final_columns)
    return feature_rows, panel


def pooled_basis(model, tokenizer, config):
    deltas = []
    for text in config["pooled_targets"]:
        target = token_id(tokenizer, text)
        panel = paired_panel(model, config["discovery"], 11, target, config["batch_size"])
        deltas.append((panel[4] - panel[1]).float())
    _, _, vh = torch.linalg.svd(torch.cat(deltas), full_matrices=False)
    return vh[:config["rank"]].T.contiguous()


def gate(row, oracle):
    return row["fidelity"] >= .70 and row["fidelity"] >= oracle["fidelity"] - .15


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    basis = pooled_basis(model, tokenizer, config)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_basis = torch.linalg.qr(torch.randn(model.config.n_embd, config["rank"],
                                               generator=generator, device="cuda")).Q
    train_features, train_panel = collect(
        model, config["discovery"], target, config["batch_size"], config["sequence_length"],
    )
    train_coefficients = ((train_panel[4] - train_panel[1]).float() @ basis).cpu()
    fitted = {
        depth: {
            family: fit_ridge(train_features[depth][family], train_coefficients,
                              config["ridge_relative_penalty"])
            for family in config["feature_families"]
        }
        for depth in DEPTHS
    }

    results = {}
    maximum_trace_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        features, panel = collect(model, spec, target, config["batch_size"], config["sequence_length"])
        exact_delta = (panel[4] - panel[1]).float()
        target_coefficients = (exact_delta @ basis).cpu()
        oracle = causal(model, panel, (target_coefficients.to("cuda") @ basis.T))
        random_oracle = causal(model, panel, (exact_delta @ random_basis) @ random_basis.T)
        native_panel = paired_panel(model, spec, 11, target, config["batch_size"])
        maximum_trace_error = max(maximum_trace_error, float((panel[2] - native_panel[2]).abs().max()))
        depths = {}
        permutation = torch.randperm(len(target_coefficients), generator=torch.Generator().manual_seed(
            config["seed"] + sum(map(ord, corpus))))
        for depth in DEPTHS:
            families = {}
            for family in config["feature_families"]:
                coefficients = predict(fitted[depth][family], features[depth][family])
                predicted = causal(model, panel, coefficients.to("cuda") @ basis.T)
                permuted = causal(model, panel, coefficients[permutation].to("cuda") @ basis.T)
                families[family] = {
                    **coefficient_metrics(target_coefficients, coefficients),
                    "fidelity": predicted["fidelity"], "cosine": predicted["cosine"],
                    "permuted_fidelity": permuted["fidelity"],
                    "permutation_drop": predicted["fidelity"] - permuted["fidelity"],
                }
            depths[str(depth)] = families
        results[corpus] = {"examples": len(target_coefficients), "oracle": oracle,
                           "random_oracle": random_oracle, "depths": depths}

    best = {
        corpus: [max(results[corpus]["depths"][str(depth)].items(),
                     key=lambda item: item[1]["fidelity"])
                 for depth in DEPTHS]
        for corpus in results
    }
    universal = []
    for depth in DEPTHS:
        for family in config["feature_families"]:
            if all(gate(results[c]["depths"][str(depth)][family], results[c]["oracle"])
                   for c in results):
                universal.append((depth, family))
    earliest = min(universal) if universal else None
    fineweb_first = next(((depth, family) for depth in DEPTHS
                          for family in config["feature_families"]
                          if gate(results["fineweb"]["depths"][str(depth)][family],
                                  results["fineweb"]["oracle"])), None)
    r0 = maximum_trace_error <= 1e-5 and all(
        row["oracle"]["fidelity"] >= row["random_oracle"]["fidelity"] + .30
        for row in results.values())
    r1 = any(depth == 10 for depth, _ in universal)
    r2 = earliest is not None and earliest[0] <= 6
    r3 = earliest is not None and all(
        results[c]["depths"][str(earliest[0])][earliest[1]]["permutation_drop"] >= .15
        for c in results)
    r4 = all(any(results[c]["depths"][str(depth)]["polynomial"]["fidelity"]
                     >= results[c]["depths"][str(depth)]["delta"]["fidelity"] + .10
                     for depth in DEPTHS) for c in results)
    correlations = {c: spearman([row[1]["fidelity"] for row in best[c]]) for c in results}
    r5 = all(value >= .70 for value in correlations.values())
    r6 = fineweb_first is not None and all(
        gate(results[c]["depths"][str(fineweb_first[0])][fineweb_first[1]], results[c]["oracle"])
        for c in ("wikitext", "tinystories"))
    hypotheses = {
        "R0_trace_and_basis_are_valid": {"supported": r0, "maximum_trace_error": maximum_trace_error},
        "R1_pre_final_local_router_is_sufficient": {"supported": r1},
        "R2_mid_depth_router_is_sufficient": {"supported": r2, "earliest_universal": earliest},
        "R3_context_specific_coefficients_matter": {"supported": r3},
        "R4_nonlinear_local_features_add_value": {"supported": r4},
        "R5_router_quality_generally_increases_with_depth": {"supported": r5,
                                                               "correlations": correlations},
        "R6_one_frozen_router_transfers_ood": {"supported": r6,
                                                 "fineweb_first": fineweb_first},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for corpus in results:
        axes[0].plot(DEPTHS, [max(results[corpus]["depths"][str(depth)][family]["fidelity"]
                                      for family in config["feature_families"])
                              for depth in DEPTHS], marker="o", label=corpus)
        axes[1].plot(DEPTHS, [results[corpus]["depths"][str(depth)]["delta"]["fidelity"]
                              for depth in DEPTHS], marker="o", label=corpus)
        axes[2].plot(DEPTHS, [results[corpus]["depths"][str(depth)]["polynomial"]["permutation_drop"]
                              for depth in DEPTHS], marker="o", label=corpus)
    axes[0].set_title("Best frozen local router"); axes[1].set_title("Delta-only router")
    axes[2].set_title("Polynomial minus permuted coefficients")
    for axis in axes:
        axis.axhline(.70 if axis is not axes[2] else .15, color="black", ls="--", lw=.8)
        axis.set_xlabel("boundary (-1 = embedding)"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Depth of the compact late-write router", "",
        "A pooled FineWeb rank-32 final-MLP write basis is frozen across four token edits. For token-to-newline edits, ridge maps trained only on FineWeb predict its coefficients from the embedding boundary and residual boundaries after Layers 0--10. Every prediction is evaluated by a finite write intervention in the original final layer.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Depth router](bilin12_backbone_router_depth.png)", "",
              "| Corpus | Oracle/random fidelity | Earliest passing depth/family | Best fidelity at embedding / L3 / L6 / L10 | L10 winning family |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        passing = [(depth, family) for depth in DEPTHS for family in config["feature_families"]
                   if gate(row["depths"][str(depth)][family], row["oracle"])]
        summaries = []
        for depth in (-1, 3, 6, 10):
            family, value = max(row["depths"][str(depth)].items(), key=lambda item: item[1]["fidelity"])
            summaries.append(f"{value['fidelity']:.3f} ({family})")
        winner, _ = max(row["depths"]["10"].items(), key=lambda item: item[1]["fidelity"])
        lines.append(f"| {corpus} | {row['oracle']['fidelity']:.3f}/{row['random_oracle']['fidelity']:.3f} | "
                     f"{passing[0] if passing else 'none'} | {' / '.join(summaries)} | {winner} |")
    lines += ["", "A successful local router is a readable interface, not proof that the model itself uses that coordinate system. A failure is also scoped: it rejects these frozen low-complexity destination-state maps, not all possible distributed or source-aware explanations. Permuting predicted coefficients within corpus tests whether per-example routing adds causal value beyond a generic mean edit."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
