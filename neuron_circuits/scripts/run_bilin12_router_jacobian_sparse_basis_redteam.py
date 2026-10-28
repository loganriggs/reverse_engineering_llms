#!/usr/bin/env python3
"""Coverage-controlled ICA/varimax red-team of Jacobian mode semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from discover_bilin18_weight_only_circuits import auc
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_jacobian_mode_semantics import collect, make_labels, vocabulary_labels
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_jacobian_sparse_basis_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_jacobian_sparse_basis_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JACOBIAN_SPARSE_BASIS_REDTEAM.md"


def robust_labels(train, evaluations, config):
    count = train.sum(0); valid = (count >= config["minimum_train_positive"]) & (count <= len(train) - config["minimum_train_positive"])
    for labels in evaluations.values():
        count = labels.sum(0)
        valid &= (count >= config["minimum_eval_positive"]) & (count <= len(labels) - config["minimum_eval_positive"])
    return np.flatnonzero(valid)


def select(scores, labels, names, valid):
    rows = []
    for mode in range(scores.shape[1]):
        candidates = []
        for label in valid:
            value = auc(labels[:, label], scores[:, mode])
            candidates.append((max(value, 1 - value), int(label), 1 if value >= .5 else -1))
        best, label, sign = max(candidates)
        rows.append({"mode": mode, "label_index": label, "label": names[label], "sign": sign, "discovery_auc": best})
    return rows


def evaluate(scores, labels, selection):
    values = [auc(labels[:, row["label_index"]], row["sign"] * scores[:, row["mode"]]) for row in selection]
    return {"median_auc": float(np.median(values)), "modes_above_080": int((np.asarray(values) >= .8).sum()),
            "mode_aucs": values}


def symmetric_decorrelation(weight):
    values, vectors = torch.linalg.eigh(weight @ weight.T)
    return (vectors * values.clamp_min(1e-12).rsqrt()[None]) @ vectors.T @ weight


def fast_ica(x, generator, iterations):
    dimension = x.shape[1]
    weight = symmetric_decorrelation(torch.randn(dimension, dimension, generator=generator))
    for step in range(iterations):
        old = weight; y = x @ weight.T; g = torch.tanh(y); gp = 1 - g.square()
        weight = (g.T @ x) / len(x) - gp.mean(0)[:, None] * weight
        weight = symmetric_decorrelation(weight)
        convergence = float((1 - (weight @ old.T).diagonal().abs()).abs().max())
        if convergence < 1e-6: break
    q = weight.T
    components = x @ q
    kurtosis = float((components.pow(4).mean(0) - 3).abs().mean())
    return q, {"iterations": step + 1, "convergence": convergence, "absolute_excess_kurtosis": kurtosis}


def varimax(x, iterations=500, tolerance=1e-7):
    dimension = x.shape[1]; rotation = torch.eye(dimension); previous = 0.0
    for step in range(iterations):
        loadings = x @ rotation
        column_energy = loadings.square().sum(0)
        target = loadings.pow(3) - loadings * column_energy[None] / len(x)
        u, singular, vh = torch.linalg.svd(x.T @ target, full_matrices=False)
        rotation = u @ vh; objective = float(singular.sum())
        if previous and objective <= previous * (1 + tolerance): break
        previous = objective
    return rotation, {"iterations": step + 1, "objective": previous,
                      "absolute_excess_kurtosis": float(((x @ rotation).pow(4).mean(0) - 3).abs().mean())}


def greedy_overlap(first, second):
    overlap = (first.T @ second).abs().numpy(); used_left = set(); used_right = set(); selected = []
    for flat in np.argsort(overlap, axis=None)[::-1]:
        left, right = np.unravel_index(flat, overlap.shape)
        if left not in used_left and right not in used_right:
            selected.append(overlap[left, right]); used_left.add(left); used_right.add(right)
            if len(selected) == overlap.shape[0]: break
    return float(np.mean(selected))


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
    vocabulary, target_ids = vocabulary_labels(tokenizer, config["targets"])
    print("collect discovery", flush=True); matrix, current, next_ids, positions = collect(model, config["mode_train"], config, axes)
    train_labels, names = make_labels(current, next_ids, positions, vocabulary, target_ids, config)
    mean = matrix.mean(0); centered = matrix - mean; _, _, vh = torch.linalg.svd(centered.cuda(), full_matrices=False)
    modes = vh[:config["rank"]].cpu(); raw_scores = centered @ modes.T
    score_scale = raw_scores.std(0).clamp_min(1e-6); whitened = raw_scores / score_scale
    evaluation_data = {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True); value, current, next_ids, positions = collect(model, spec, config, axes)
        labels, _ = make_labels(current, next_ids, positions, vocabulary, target_ids, config)
        evaluation_data[corpus] = {"scores": ((value - mean) @ modes.T) / score_scale, "labels": labels}
    valid = robust_labels(train_labels, {key: row["labels"] for key, row in evaluation_data.items()}, config)
    generator = torch.Generator().manual_seed(config["seed"])
    ica_runs = [fast_ica(whitened, generator, config["ica_iterations"]) for _ in range(config["ica_starts"])]
    best_index = int(np.argmax([metadata["absolute_excess_kurtosis"] for _, metadata in ica_runs])); ica_q, ica_metadata = ica_runs[best_index]
    ica_metadata["selected_start"] = best_index
    overlaps = [greedy_overlap(ica_q, q) for index, (q, _) in enumerate(ica_runs) if index != best_index]
    varimax_q, varimax_metadata = varimax(whitened)
    bases = {"pca": torch.eye(config["rank"]), "ica": ica_q, "varimax": varimax_q}
    selections = {name: select((whitened @ q).numpy(), train_labels, names, valid) for name, q in bases.items()}
    random_rows = []
    for _ in range(config["random_rotations"]):
        q, _ = torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator))
        random_rows.append({"q": q, "selection": select((whitened @ q).numpy(), train_labels, names, valid)})
    results = {}
    for corpus, row in evaluation_data.items():
        values = {name: evaluate((row["scores"] @ q).numpy(), row["labels"], selections[name]) for name, q in bases.items()}
        random_medians = [evaluate((row["scores"] @ item["q"]).numpy(), row["labels"], item["selection"])["median_auc"] for item in random_rows]
        values["random_rotation_medians"] = random_medians; values["random_p95"] = float(np.quantile(random_medians, .95))
        results[corpus] = values
    r0 = all(row["pca"]["median_auc"] >= .70 for row in results.values())
    r1 = all(row["ica"]["median_auc"] >= row["pca"]["median_auc"] + .05 for row in results.values())
    r2 = all(row["ica"]["median_auc"] >= row["random_p95"] + .05 for row in results.values())
    r3 = all(row["varimax"]["median_auc"] >= row["pca"]["median_auc"] + .05 for row in results.values())
    r4 = all(value >= .80 for value in overlaps)
    better = "ica" if np.mean([row["ica"]["median_auc"] for row in results.values()]) >= np.mean([row["varimax"]["median_auc"] for row in results.values()]) else "varimax"
    diversity = len({row["label"] for row in selections[better]}); r5 = diversity >= 5
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "R0_pca_result_survives_coverage_control": r0, "R1_ica_improves_individual_semantics": r1,
        "R2_ica_is_privileged_over_random_rotations": r2, "R3_varimax_improves_individual_semantics": r3,
        "R4_ica_coordinates_are_reproducible": r4, "R5_sparse_basis_labels_are_diverse": r5}.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "robust_label_indices": valid.tolist(),
                                  "robust_label_names": [names[index] for index in valid], "ica": ica_metadata,
                                  "ica_cross_start_overlap": overlaps, "varimax": varimax_metadata,
                                  "better_sparse_basis": better, "selections": selections, "results": results}, indent=2, allow_nan=False))
    lines = ["# Sparse-basis red-team of Jacobian semantics", "",
             f"Only {len(valid)} predicates with adequate support on discovery FineWeb and every evaluation corpus are eligible. Top-16 PCA scores are whitened before ICA, varimax, and 20 matched random rotations. ICA start selection uses excess kurtosis only, never semantic labels.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | PCA | ICA | Varimax | Random p95 |", "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items(): lines.append(f"| {corpus} | {row['pca']['median_auc']:.3f} | {row['ica']['median_auc']:.3f} | {row['varimax']['median_auc']:.3f} | {row['random_p95']:.3f} |")
    lines += ["", f"ICA cross-start greedy overlaps: `{[round(value, 3) for value in overlaps]}`. Better sparse basis by mean held-out median: **{better}**; distinct selected predicates: **{diversity}**.", "",
              "A sparse rotation that transfers and beats random rotations would supply a more canonical verb basis than PCA. If sparse rotations improve but do not beat random controls or reproduce across starts, they are useful coordinate choices rather than uniquely identified model objects."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
