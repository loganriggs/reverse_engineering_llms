#!/usr/bin/env python3
"""Validate a signed margin-constraint explanation on factual checkpoints.

The design and thresholds are locked in
artifacts/constraint_network_preregistration.md.  This script deliberately
uses existing checkpoints and writes into a new artifact directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from neuron_circuits.factual_data import ATTRIBUTES, VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.factual_model import FactualTransformer


ROOT = Path("artifacts/factual_capacity_calibration_r2")
OUT = Path("artifacts/constraint_network_pilot")
WIDTHS = (128, 256)
RHOS = (0.0, 0.5, 1.0)
SCALES = (0.25, 1.0, 4.0)
PANEL_SIZE = 32
SOURCE_COUNT = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--source-count", type=int, default=SOURCE_COUNT)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--rhos", type=float, nargs="+", default=RHOS)
    return parser.parse_args()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: Path, device: str) -> FactualTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FactualTransformer(VOCAB.size, checkpoint["width"], checkpoint["heads"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def prompts_for_indices(manifest: dict, indices: np.ndarray, template: int = 0) -> np.ndarray:
    entities = indices // ATTRIBUTES
    attributes = (indices % ATTRIBUTES).astype(np.uint8)
    return encode_prompts(manifest["ids"][entities], manifest["traits"][entities], attributes, template)


@torch.no_grad()
def logits_for(model: FactualTransformer, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens).float()


def fixed_margins(logits: torch.Tensor, targets: torch.Tensor, competitors: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(targets), device=logits.device)
    return logits[rows, targets] - logits[rows, competitors]


def strongest_competitors(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    wrong = logits.clone()
    wrong[torch.arange(len(targets), device=logits.device), targets] = -torch.inf
    return wrong.argmax(1)


def parameter_groups(model: FactualTransformer):
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if name == "position" or name.startswith("embedding."):
            group = "input"
        elif name.startswith("blocks."):
            group = f"block_{name.split('.')[1]}"
        else:
            group = "output"
        groups[group].append((name, parameter))
    order = ["input", *(f"block_{i}" for i in range(model.layers_count)), "output"]
    return {name: groups[name] for name in order}


def margin_gradients(model, tokens, targets, competitors, groups):
    """Return exact float32 per-example gradient matrices on CPU."""
    all_parameters = [parameter for group in groups.values() for _, parameter in group]
    slices = {}
    cursor = 0
    for group_name, entries in groups.items():
        count = len(entries)
        slices[group_name] = slice(cursor, cursor + count)
        cursor += count
    matrices = {
        name: torch.empty((len(tokens), sum(p.numel() for _, p in entries)), dtype=torch.float32)
        for name, entries in groups.items()
    }
    for row in range(len(tokens)):
        logits = model(tokens[row:row + 1]).float()
        margin = logits[0, targets[row]] - logits[0, competitors[row]]
        gradients = torch.autograd.grad(margin, all_parameters)
        for group_name, entries in groups.items():
            offset = 0
            for gradient in gradients[slices[group_name]]:
                flat = gradient.detach().float().reshape(-1).cpu()
                matrices[group_name][row, offset:offset + len(flat)] = flat
                offset += len(flat)
    return matrices


def stratified_take(candidates: np.ndarray, labels: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in candidates:
        buckets[(int(index % ATTRIBUTES), int(labels[index]))].append(int(index))
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets)
    chosen = []
    depth = 0
    while len(chosen) < count:
        added = False
        for key in keys:
            if depth < len(buckets[key]):
                chosen.append(buckets[key][depth]); added = True
                if len(chosen) == count:
                    break
        if not added:
            raise RuntimeError(f"only found {len(chosen)} eligible facts, requested {count}")
        depth += 1
    return chosen


def shared_panel(root: Path, manifest: dict, rho: float, size: int, widths: tuple[int, ...]) -> np.ndarray:
    labels, selected, changed = labels_for_rho(manifest, rho)
    labels_flat, selected_flat, changed_flat = labels.reshape(-1), selected.reshape(-1), changed.reshape(-1)
    eligible = None
    for width in widths:
        with np.load(root / f"rho_{rho:g}" / f"width_{width}" / "predictions.npz") as stored:
            indices = stored["seen_fact_indices"]
            correct = ((stored["seen_fact_template0_prediction"] == stored["seen_fact_targets"])
                       & (stored["seen_fact_template6_prediction"] == stored["seen_fact_targets"]))
            current = set(indices[correct].tolist())
            eligible = current if eligible is None else eligible & current
    candidate = np.asarray(sorted(eligible), dtype=np.int64)
    rng = np.random.default_rng(20261017 + round(100 * rho))
    if rho == 0.5:
        random_candidates = candidate[changed_flat[candidate]]
        structured_candidates = candidate[~selected_flat[candidate]]
        chosen = stratified_take(random_candidates, labels_flat, size // 2, rng)
        chosen += stratified_take(structured_candidates, labels_flat, size - size // 2, rng)
    elif rho == 1.0:
        chosen = stratified_take(candidate[changed_flat[candidate]], labels_flat, size, rng)
    else:
        chosen = stratified_take(candidate, labels_flat, size, rng)
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def origin_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(actual ** 2))
    return float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 0 else float("nan")


def kernel_summary(cosine: np.ndarray, metadata: dict[str, np.ndarray]) -> dict:
    n = len(cosine)
    keep = ~np.eye(n, dtype=bool)
    values = cosine[keep]
    supports = []
    for row in range(n):
        edges = np.abs(np.delete(cosine[row], row))
        supports.append(float(edges.sum() ** 2 / max(np.sum(edges ** 2), 1e-30)))
    pair_i, pair_j = np.triu_indices(n, 1)
    y = cosine[pair_i, pair_j]
    features = np.column_stack((
        np.ones(len(pair_i)),
        metadata["attribute"][pair_i] == metadata["attribute"][pair_j],
        metadata["answer"][pair_i] == metadata["answer"][pair_j],
        metadata["competitor"][pair_i] == metadata["competitor"][pair_j],
        metadata["selected"][pair_i] == metadata["selected"][pair_j],
        (metadata["traits"][pair_i] == metadata["traits"][pair_j]).mean(1),
        (metadata["ids"][pair_i] == metadata["ids"][pair_j]).mean(1),
    )).astype(np.float64)
    coefficients = np.linalg.lstsq(features, y, rcond=None)[0]
    residual = y - features @ coefficients
    return {
        "mean_absolute_cosine": float(np.mean(np.abs(values))),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "mean_effective_support": float(np.mean(supports)),
        "control_r2": float(1 - np.sum(residual ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)),
        "mean_absolute_residual_cosine": float(np.mean(np.abs(residual))),
        "control_coefficients": coefficients.tolist(),
    }


def choose_sources(metadata: dict[str, np.ndarray], count: int) -> np.ndarray:
    # Round-robin over randomized status, attribute, and answer.  This is
    # deterministic and independent of gradients or edit outcomes.
    keys = [(int(metadata["selected"][i]), int(metadata["attribute"][i]), int(metadata["answer"][i]), i)
            for i in range(len(metadata["answer"]))]
    ordered = [row[-1] for row in sorted(keys)]
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return np.asarray([ordered[p] for p in positions], dtype=np.int64)


def finite_edits(model, tokens, targets, competitors, baseline_logits, groups, grams, sources):
    baseline_margin = fixed_margins(baseline_logits, targets, competitors).detach().cpu().numpy()
    all_parameters = [parameter for entries in groups.values() for _, parameter in entries]
    group_slices = {}
    cursor = 0
    for name, entries in groups.items():
        group_slices[name] = slice(cursor, cursor + len(entries)); cursor += len(entries)
    records = []
    for source in sources:
        logits = model(tokens[source:source + 1]).float()
        margin = logits[0, targets[source]] - logits[0, competitors[source]]
        gradients = torch.autograd.grad(margin, all_parameters)
        for group_name, entries in groups.items():
            group_gradients = gradients[group_slices[group_name]]
            norm_sq = float(sum(torch.sum(g.detach().float() ** 2) for g in group_gradients))
            parameter_norm_sq = float(sum(torch.sum(p.detach().float() ** 2) for _, p in entries))
            if norm_sq <= 1e-20:
                continue
            originals = [p.detach().clone() for _, p in entries]
            for scale in SCALES:
                coefficient = scale / norm_sq
                with torch.no_grad():
                    for (_, parameter), gradient in zip(entries, group_gradients):
                        parameter.add_(gradient, alpha=coefficient)
                    edited_logits = logits_for(model, tokens)
                    edited_margin = fixed_margins(edited_logits, targets, competitors).cpu().numpy()
                    edited_competitor = strongest_competitors(edited_logits, targets).cpu().numpy()
                    for (_, parameter), original in zip(entries, originals):
                        parameter.copy_(original)
                actual = edited_margin - baseline_margin
                gram = grams[group_name]
                predicted = scale * gram[:, source] / gram[source, source]
                null_source = int(sources[(np.flatnonzero(sources == source)[0] + 1) % len(sources)])
                null_predicted = scale * gram[:, null_source] / gram[null_source, null_source]
                keep = np.arange(len(tokens)) != source
                records.append({
                    "source": int(source), "group": group_name, "scale": scale,
                    "actual": actual.tolist(), "predicted": predicted.tolist(),
                    "null_predicted": null_predicted.tolist(),
                    "offdiag_pearson": corr(actual[keep], predicted[keep]),
                    "offdiag_origin_r2": origin_r2(actual[keep], predicted[keep]),
                    "null_pearson": corr(actual[keep], null_predicted[keep]),
                    "null_origin_r2": origin_r2(actual[keep], null_predicted[keep]),
                    "source_actual": float(actual[source]),
                    "competitor_switch_fraction": float(np.mean(edited_competitor != competitors.cpu().numpy())),
                    "relative_parameter_change": float((scale / np.sqrt(norm_sq)) / max(np.sqrt(parameter_norm_sq), 1e-30)),
                })
    return records


def pooled_edit_metrics(records: list[dict], group_names: list[str], scale: float) -> dict:
    selected = [row for row in records if row["group"] in group_names and row["scale"] == scale]
    actual, predicted, null = [], [], []
    collateral = []
    for row in selected:
        a = np.asarray(row["actual"]); p = np.asarray(row["predicted"]); q = np.asarray(row["null_predicted"])
        keep = np.arange(len(a)) != row["source"]
        actual.extend(a[keep]); predicted.extend(p[keep]); null.extend(q[keep])
        collateral.append(float(np.mean(np.abs(a[keep]))))
    actual, predicted, null = map(np.asarray, (actual, predicted, null))
    return {
        "count": len(actual), "pearson": corr(actual, predicted),
        "origin_r2": origin_r2(actual, predicted), "null_pearson": corr(actual, null),
        "null_origin_r2": origin_r2(actual, null),
        "mean_absolute_collateral": float(np.mean(collateral)),
    }


def run_condition(args, manifest, rho: float, width: int, panel: np.ndarray) -> dict:
    started = time.perf_counter()
    model = load_model(args.root / f"rho_{rho:g}" / f"width_{width}" / "checkpoint.pt", args.device)
    labels, selected, changed = labels_for_rho(manifest, rho)
    targets = torch.from_numpy(labels.reshape(-1)[panel].astype(np.int64)).to(args.device)
    tokens = torch.from_numpy(prompts_for_indices(manifest, panel).astype(np.int64)).to(args.device)
    baseline_logits = logits_for(model, tokens)
    competitors = strongest_competitors(baseline_logits, targets)
    baseline_accuracy = float((baseline_logits.argmax(1) == targets).float().mean())
    entities = panel // ATTRIBUTES
    metadata = {
        "attribute": panel % ATTRIBUTES,
        "answer": targets.cpu().numpy(),
        "competitor": competitors.cpu().numpy(),
        "selected": selected.reshape(-1)[panel],
        "changed": changed.reshape(-1)[panel],
        "traits": manifest["traits"][entities],
        "ids": manifest["ids"][entities],
    }
    groups = parameter_groups(model)
    gradient_matrices = margin_gradients(model, tokens, targets, competitors, groups)
    grams, kernels = {}, {}
    for group_name, matrix in gradient_matrices.items():
        gram = (matrix @ matrix.T).numpy()
        norm = np.sqrt(np.maximum(np.diag(gram), 1e-30))
        cosine = gram / norm[:, None] / norm[None, :]
        grams[group_name] = gram
        kernels[group_name] = {"summary": kernel_summary(cosine, metadata), "cosine": cosine.tolist()}
    del gradient_matrices
    sources = choose_sources(metadata, args.source_count)
    edits = finite_edits(model, tokens, targets, competitors, baseline_logits, groups, grams, sources)
    block_names = [f"block_{i}" for i in range(model.layers_count)]
    pooled = {str(scale): pooled_edit_metrics(edits, block_names, scale) for scale in SCALES}
    return {
        "rho": rho, "width": width, "panel_indices": panel.tolist(), "sources": sources.tolist(),
        "baseline_accuracy": baseline_accuracy,
        "metadata": {key: value.tolist() for key, value in metadata.items()},
        "kernels": kernels, "edits": edits, "pooled_blocks": pooled,
        "elapsed_seconds": time.perf_counter() - started,
    }


def verdict(results: list[dict], widths: tuple[int, ...]) -> dict:
    full_random = [row for row in results if row["rho"] == 1.0]
    decisions = {}
    for scale in (0.25, 1.0):
        metrics = [row["pooled_blocks"][str(scale)] for row in full_random]
        passed = all(m["pearson"] >= .8 and m["origin_r2"] >= .5
                     and m["pearson"] - m["null_pearson"] >= .3 for m in metrics)
        falsified = any(m["pearson"] < .3 or m["pearson"] - m["null_pearson"] < .1 for m in metrics)
        decisions[str(scale)] = "validated" if passed else "falsified" if falsified else "inconclusive"
    load_rows = {}
    for width in widths:
        low = next(row for row in results if row["width"] == width and row["rho"] == 0.0)
        high = next(row for row in results if row["width"] == width and row["rho"] == 1.0)
        low_residual = np.mean([low["kernels"][f"block_{i}"]["summary"]["mean_absolute_residual_cosine"] for i in range(4)])
        high_residual = np.mean([high["kernels"][f"block_{i}"]["summary"]["mean_absolute_residual_cosine"] for i in range(4)])
        low_collateral = low["pooled_blocks"]["1.0"]["mean_absolute_collateral"]
        high_collateral = high["pooled_blocks"]["1.0"]["mean_absolute_collateral"]
        load_rows[str(width)] = {
            "rho0_residual_cosine": float(low_residual), "rho1_residual_cosine": float(high_residual),
            "residual_increased": bool(high_residual > low_residual),
            "rho0_collateral": low_collateral, "rho1_collateral": high_collateral,
            "collateral_increased": bool(high_collateral > low_collateral),
        }
    web_supported = all(row["residual_increased"] or row["collateral_increased"] for row in load_rows.values())
    return {
        "local_constraint_explanation": decisions["0.25"],
        "finite_constraint_explanation": decisions["1.0"],
        "load_induced_web": "supported" if web_supported else "falsified",
        "load_comparisons": load_rows,
    }


def write_summary(output: Path, payload: dict) -> None:
    lines = [
        "# Signed factual constraint-network pilot", "",
        "The design and thresholds were locked before these kernels or edits were computed. See `../constraint_network_preregistration.md`.", "",
        "## Primary causal results", "",
        "Pooled scores use all four transformer blocks and exclude each edit's source fact.", "",
        "| rho | width | scale | Pearson | origin R² | null Pearson | mean absolute collateral | rival switches |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        for scale in SCALES:
            metric = run["pooled_blocks"][str(scale)]
            selected = [row for row in run["edits"] if row["group"].startswith("block_") and row["scale"] == scale]
            switches = np.mean([row["competitor_switch_fraction"] for row in selected])
            lines.append(f"| {run['rho']:g} | {run['width']} | {scale:g} | {metric['pearson']:.3f} | {metric['origin_r2']:.3f} | {metric['null_pearson']:.3f} | {metric['mean_absolute_collateral']:.5f} | {switches:.3f} |")
    lines.extend(("", "## Preregistered verdict", ""))
    for key, value in payload["verdict"].items():
        if key != "load_comparisons":
            lines.append(f"- {key.replace('_', ' ').capitalize()}: **{value}**")
    lines.extend(("", "## Load comparison", "",
                  "Values average the four blockwise controlled residual-cosine summaries; collateral uses 1-logit edits.", "",
                  "| width | rho=0 residual | rho=1 residual | rho=0 collateral | rho=1 collateral |",
                  "|---:|---:|---:|---:|---:|"))
    for width, row in payload["verdict"]["load_comparisons"].items():
        lines.append(f"| {width} | {row['rho0_residual_cosine']:.5f} | {row['rho1_residual_cosine']:.5f} | {row['rho0_collateral']:.5f} | {row['rho1_collateral']:.5f} |")
    lines.extend(("", "## Scope", "",
                  "A successful local Taylor prediction validates the signed constraint network as a causal local model, not as a unique or complete account of storage. The load test asks the stronger empirical question: whether arbitrary facts make that network broader after controlling for obvious shared fields. Semantic coexistence must be assessed with the locked activation audit and a separate transfer intervention; it is not inferred from gradient coupling alone.", ""))
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    torch.manual_seed(20261017); np.random.seed(20261017)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.root / "manifest.npz")
    widths, rhos = tuple(args.widths), tuple(args.rhos)
    if 0.0 not in rhos or 1.0 not in rhos:
        raise ValueError("the preregistered verdict requires rho 0 and 1")
    panels = {rho: shared_panel(args.root, manifest, rho, args.panel_size, widths) for rho in rhos}
    dump(args.output / "panels.json", {str(rho): panel.tolist() for rho, panel in panels.items()})
    runs = []
    for rho in rhos:
        for width in widths:
            print(f"START rho={rho:g} width={width}", flush=True)
            row = run_condition(args, manifest, rho, width, panels[rho])
            runs.append(row)
            dump(args.output / f"rho_{rho:g}_width_{width}.json", row)
            print(f"DONE rho={rho:g} width={width} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    payload = {"design": {"panel_size": args.panel_size, "source_count": args.source_count,
                           "widths": widths, "rhos": rhos, "scales": SCALES}, "runs": runs}
    payload["verdict"] = verdict(runs, widths)
    dump(args.output / "results.json", payload)
    write_summary(args.output, payload)
    print(json.dumps(payload["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
