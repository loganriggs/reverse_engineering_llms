#!/usr/bin/env python3
"""Red-team semantic carrier localization with equal-budget spatial masks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_spatial_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_spatial_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_SPATIAL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_spatial_redteam.png"


@torch.inference_mode()
def trace_model(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    outputs = []
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
        outputs.append(x)
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)
    return x0, value_bus, outputs, logits


@torch.inference_mode()
def suffix(model, layer, states, x0, value_bus):
    dimension = model.config.n_embd
    x = states
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)


def transplant(base, changed, mask):
    result = base.clone()
    result[mask] = changed[mask]
    return result


def masks_for_batch(length, cue, boundary, window_sizes, device):
    batch = len(cue)
    positions = torch.arange(length, device=device)[None, :]
    rows = torch.arange(batch, device=device)[:, None]
    cue2, boundary2 = cue[:, None], boundary[:, None]
    masks = {
        "full": torch.ones((batch, length), dtype=torch.bool, device=device),
        "pre_boundary": (positions >= cue2) & (positions < boundary2),
        "post_boundary": positions >= boundary2,
        "post_boundary_no_destination": (positions >= boundary2) & (positions < length - 1),
    }
    # All matched controls are restricted to the causally eligible cue-to-destination field.
    eligible_count = length - cue
    for k in window_sizes:
        count = torch.minimum(torch.full_like(eligible_count, k), eligible_count)
        masks[f"last_{k}"] = positions >= (length - count[:, None])
        masks[f"first_{k}"] = (positions >= cue2) & (positions < cue2 + count[:, None])
        # Deterministic approximately-uniform samples over the same eligible field.
        rank = torch.arange(k, device=device)[None, :].expand(batch, -1)
        denom = torch.maximum(count[:, None] - 1, torch.ones_like(count[:, None]))
        span = torch.maximum(eligible_count[:, None] - 1, torch.zeros_like(eligible_count[:, None]))
        selected = cue2 + torch.div(rank * span, denom, rounding_mode="floor")
        valid = rank < count[:, None]
        spread = torch.zeros((batch, length), dtype=torch.bool, device=device)
        spread[rows.expand_as(selected)[valid], selected[valid]] = True
        masks[f"spread_{k}"] = spread
    return masks


def metrics(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return {
        "mean": float(value.mean()),
        "projection_share": float((value * target).sum() / energy),
        "energy_fidelity": float(1 - (target - value).square().sum() / energy),
        "direction_agreement": float((value * target > 0).double().mean()),
        "pearson": float(np.corrcoef(target.numpy(), value.numpy())[0, 1]) if float(value.std()) > 0 else 0.0,
    }


@torch.inference_mode()
def collect_cell(model, raw, records, contrast, layers, window_sizes, batch_size, program_chunk_size):
    names = ["full", "pre_boundary", "post_boundary", "post_boundary_no_destination"]
    for k in window_sizes:
        names.extend((f"last_{k}", f"first_{k}", f"spread_{k}"))
    stored = {layer: {name: [] for name in names} for layer in layers}
    targets = {layer: [] for layer in layers}
    position_rows = []
    maximum_error = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        changed = original.copy()
        for index, row in enumerate(batch):
            changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        x0, bus, outputs, native_logits = trace_model(model, ids)
        size = len(batch)
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        position_rows.extend({"cue": int(c), "boundary": int(b), "post_count": 128 - int(b)} for c, b in zip(cue.cpu(), boundary.cpu()))
        native_target = direction * ((native_logits[size:] @ contrast) - (native_logits[:size] @ contrast))
        spatial_masks = masks_for_batch(128, cue, boundary, window_sizes, ids.device)
        for layer in layers:
            base_state, changed_state = outputs[layer][:size], outputs[layer][size:]
            base_logits = suffix(
                model,
                layer,
                base_state,
                x0[:size],
                bus[:size],
            )
            base_score = base_logits @ contrast
            targets[layer].append(native_target.cpu())
            maximum_error = max(maximum_error, float((base_logits - native_logits[:size]).abs().max()))
            for chunk_start in range(0, len(names), program_chunk_size):
                chunk = names[chunk_start:chunk_start + program_chunk_size]
                states = [transplant(base_state, changed_state, spatial_masks[name]) for name in chunk]
                logits = suffix(
                    model,
                    layer,
                    torch.cat(states),
                    torch.cat([x0[:size]] * len(states)),
                    torch.cat([bus[:size]] * len(states)),
                ).reshape(len(states), size, -1)
                for index, name in enumerate(chunk):
                    stored[layer][name].append((direction * ((logits[index] @ contrast) - base_score)).cpu())
        print(f"spatial batch {min(start + size, len(records))}/{len(records)}", flush=True)
    result = {"positions": position_rows, "layers": {}}
    for layer in layers:
        target = torch.cat(targets[layer])
        result["layers"][str(layer)] = {"examples": len(target), "full_mean": float(target.mean()), "programs": {}}
        for name in names:
            result["layers"][str(layer)]["programs"][name] = metrics(target, torch.cat(stored[layer][name]))
    return result, maximum_error


def summarize(results, families, layers, names):
    summary = {}
    for family in families:
        rows = [row for row in results if row["family"] == family]
        summary[family] = {}
        for layer in layers:
            key = str(layer)
            summary[family][key] = {"programs": {}}
            for name in names:
                summary[family][key]["programs"][name] = {
                    metric: float(np.mean([row["layers"][key]["programs"][name][metric] for row in rows]))
                    for metric in ("projection_share", "energy_fidelity", "direction_agreement", "pearson")
                }
    return summary


def main():
    config = json.loads(CONFIG.read_text())
    carrier_config = json.loads(Path(config["source_config"]).read_text())
    base = json.loads(Path(carrier_config["source_config"]).read_text())
    family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}
    family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    family_pairs = {name: family_pairs[name] for name in config["families"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results, maximum_error = [], 0.0
    for family_index, (family, pairs) in enumerate(family_pairs.items()):
        midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
                records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + 1000 * family_index + 10 * fold + split)
                print(f"spatial cell {family} fold={fold} split={split} n={len(records)}", flush=True)
                cell, error = collect_cell(model, raw, records, contrast, config["layers"], config["window_sizes"], config["batch_size"], config["program_chunk_size"])
                maximum_error = max(maximum_error, error)
                cell.update({"family": family, "fold": fold, "split": split})
                results.append(cell)
                torch.cuda.empty_cache()
    names = ["full", "pre_boundary", "post_boundary", "post_boundary_no_destination"]
    for k in config["window_sizes"]: names.extend((f"last_{k}", f"first_{k}", f"spread_{k}"))
    summary = summarize(results, config["families"], config["layers"], names)
    min_windows = {}
    for family in config["families"]:
        min_windows[family] = {}
        for layer in config["layers"]:
            p = summary[family][str(layer)]["programs"]
            min_windows[family][str(layer)] = {
                str(threshold): next((k for k in config["window_sizes"] if p[f"last_{k}"]["projection_share"] >= threshold), None)
                for threshold in (.5, .8)
            }
    layer12, layer14 = "12", "14"
    late16 = {family: summary[family][layer12]["programs"]["last_16"]["projection_share"] - summary[family][layer12]["programs"]["first_16"]["projection_share"] for family in config["families"]}
    post_spread = {}
    for family in config["families"]:
        positions = [p["post_count"] for row in results if row["family"] == family for p in row["positions"]]
        k = min(config["window_sizes"], key=lambda value: abs(value - float(np.median(positions))))
        post_spread[family] = {"median_post_count": float(np.median(positions)), "matched_k": k, "advantage": summary[family][layer12]["programs"]["post_boundary"]["projection_share"] - summary[family][layer12]["programs"][f"spread_{k}"]["projection_share"]}
    hypotheses = {
        "S0_exact_recomposition": {"supported": maximum_error <= 1e-4, "observed": maximum_error},
        "S1_late_window_compression": {"supported": sum(min_windows[f][layer14]["0.8"] is not None and min_windows[f][layer14]["0.8"] <= 16 for f in config["families"]) >= 3, "observed": {f: min_windows[f][layer14]["0.8"] for f in config["families"]}},
        "S2_late_window_beats_early_window": {"supported": sum(value >= .20 for value in late16.values()) >= 3, "observed": late16},
        "S3_post_boundary_not_token_budget_artifact": {"supported": sum(row["advantage"] >= .10 for row in post_spread.values()) >= 3, "observed": post_spread},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "maximum_absolute_error": maximum_error, "minimum_last_window": min_windows, "summary": summary, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Equal-budget spatial red team of semantic carriers", "", "This reruns the natural semantic counterfactuals with matched spatial budgets. `last-k` patches the latest k causally eligible residual positions, `first-k` patches the earliest k positions beginning at the cue, and `spread-k` deterministically spreads k positions over the same cue-to-destination interval.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Spatial red team](bilin18_semantic_spatial_redteam.png)", "", "## Minimum latest-position window", "", "| Family | L8 k@50% | L8 k@80% | L12 k@50% | L12 k@80% | L14 k@50% | L14 k@80% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in config["families"]:
        values = []
        for layer in ("8", "12", "14"):
            for threshold in ("0.5", "0.8"):
                value = min_windows[family][layer][threshold]
                values.append("—" if value is None else str(value))
        lines.append("| " + family + " | " + " | ".join(values) + " |")
    lines += ["", "## Layer-12 equal-budget comparison", "", "| Family | Last 16 | First 16 | Spread 16 | Post-boundary | Post excl. destination |", "|---|---:|---:|---:|---:|---:|"]
    for family in config["families"]:
        p = summary[family][layer12]["programs"]
        lines.append(f"| {family} | {p['last_16']['projection_share']:.3f} | {p['first_16']['projection_share']:.3f} | {p['spread_16']['projection_share']:.3f} | {p['post_boundary']['projection_share']:.3f} | {p['post_boundary_no_destination']['projection_share']:.3f} |")
    lines += ["", "## Interpretation guardrail", "", "A successful latest-position window shows where a counterfactual state is sufficient for the original suffix, not a unique storage site. Window curves can mix destination compression with intermediate relay sites. The first/spread controls address token budget, but not every positional role; conclusions require the per-cell stability and held-out lexical-pair design already used here."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for axis, family in zip(axes.flat, config["families"]):
        for layer in (8, 12, 14):
            axis.plot(config["window_sizes"], [summary[family][str(layer)]["programs"][f"last_{k}"]["projection_share"] for k in config["window_sizes"]], marker="o", label=f"L{layer} last-k")
        axis.set_xscale("log", base=2); axis.axhline(.5, color="black", linestyle="--", linewidth=.8); axis.axhline(.8, color="gray", linestyle=":", linewidth=.8)
        axis.set(title=family, xlabel="Latest positions patched", ylabel="Projection share", xticks=config["window_sizes"]); axis.get_xaxis().set_major_formatter(plt.ScalarFormatter()); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
