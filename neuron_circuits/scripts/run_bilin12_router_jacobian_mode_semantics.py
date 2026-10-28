#!/usr/bin/env python3
"""Frozen semantic audit of spectral transport-Jacobian modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import analytic_position
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_jacobian_mode_semantics.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_jacobian_mode_semantics.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JACOBIAN_MODE_SEMANTICS.md"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


def generic_flags(text):
    stripped = text.strip()
    return np.asarray([
        bool(text[:1].isspace()),
        bool(stripped) and stripped.isalpha(),
        bool(stripped) and stripped.isdigit(),
        bool(stripped) and all(not char.isalnum() and not char.isspace() for char in stripped),
        len(stripped) == 1,
        bool(stripped) and stripped[0].isupper(),
    ], dtype=np.uint8)


GENERIC = ("leading_space", "alphabetic", "digit", "punctuation", "single_char", "capitalized")


def vocabulary_labels(tokenizer, targets):
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flags = np.stack([np.concatenate([token_flags(text), generic_flags(text)]) for text in texts])
    target_ids = []
    for target in targets:
        ids = tokenizer.encode(target, add_special_tokens=False)
        if len(ids) != 1: raise RuntimeError(f"target is not one token: {target!r}: {ids}")
        target_ids.append(ids[0])
    return flags, target_ids


@torch.inference_mode()
def collect(model, spec, config, axes):
    matrices = []; current_ids = []; next_ids = []; positions = []
    source = load_rows(spec)
    for start in range(0, len(source), config["batch_size"]):
        raw = np.asarray(source[start:start + config["batch_size"]], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        for position in config["positions"]:
            matrices.append(analytic_position(model, ids, position, config, axes, False).cpu())
            current_ids.extend(raw[:, position].tolist()); next_ids.extend(raw[:, position + 1].tolist())
            positions.extend([position] * len(raw))
    return torch.cat(matrices).reshape(-1, 1024), np.asarray(current_ids), np.asarray(next_ids), np.asarray(positions)


def make_labels(current, next_ids, positions, vocabulary, target_ids, config):
    token_names = list(CATEGORIES) + list(GENERIC)
    arrays = []; names = []
    for prefix, ids in (("current", current), ("next", next_ids)):
        for index, name in enumerate(token_names):
            arrays.append(vocabulary[ids, index]); names.append(f"{prefix}:{name}")
    for target, target_id in zip(config["targets"], target_ids):
        arrays.append(next_ids == target_id); names.append(f"next_exact:{target.encode('unicode_escape').decode()}")
    for position in config["positions"]:
        arrays.append(positions == position); names.append(f"position:{position}")
    return np.stack(arrays, 1).astype(np.uint8), names


def valid_labels(labels, minimum):
    positives = labels.sum(0)
    return np.flatnonzero((positives >= minimum) & (positives <= len(labels) - minimum))


def select_individual(scores, labels, names, minimum):
    valid = valid_labels(labels, minimum); selections = []
    for mode in range(scores.shape[1]):
        candidates = []
        for label in valid:
            value = auc(labels[:, label], scores[:, mode])
            candidates.append((max(value, 1 - value), int(label), 1 if value >= .5 else -1))
        best, label, sign = max(candidates)
        selections.append({"mode": mode, "label_index": label, "label": names[label],
                           "sign": sign, "discovery_auc": best})
    return selections


def evaluate_individual(scores, labels, selections):
    values = []
    for row in selections:
        label = row["label_index"]
        if labels[:, label].sum() == 0 or labels[:, label].sum() == len(labels): values.append(float("nan"))
        else: values.append(auc(labels[:, label], row["sign"] * scores[:, row["mode"]]))
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return {"mode_aucs": [None if not np.isfinite(value) else float(value) for value in values],
            "median_auc": float(np.median(finite)),
            "modes_above_080": int((finite >= .80).sum())}


def fit_subspace_decoders(scores, labels, minimum, ridge):
    valid = valid_labels(labels, minimum); mean = scores.mean(0); scale = scores.std(0).clamp_min(1e-6)
    z = (scores - mean) / scale; x = torch.cat([torch.ones(len(z), 1), z], 1)
    y = torch.from_numpy(labels[:, valid]).float(); gram = x.T @ x
    penalty = ridge * float(torch.trace(gram)) / len(gram); regularizer = torch.eye(len(gram)); regularizer[0, 0] = 0
    weight = torch.linalg.solve(gram + penalty * regularizer, x.T @ y)
    return {"valid": valid, "mean": mean, "scale": scale, "weight": weight}


def evaluate_subspace(scores, labels, decoder):
    z = (scores - decoder["mean"]) / decoder["scale"]
    x = torch.cat([torch.ones(len(z), 1), z], 1); predictions = x @ decoder["weight"]
    values = [auc(labels[:, label], predictions[:, index].numpy())
              for index, label in enumerate(decoder["valid"])
              if 0 < labels[:, label].sum() < len(labels)]
    return {"median_auc": float(np.median(values)), "maximum_auc": float(np.max(values)), "labels": len(values)}


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
    print("collect discovery", flush=True); train_matrix, current, next_ids, position = collect(model, config["mode_train"], config, axes)
    train_labels, names = make_labels(current, next_ids, position, vocabulary, target_ids, config)
    mean = train_matrix.mean(0); centered = train_matrix - mean
    _, _, vh = torch.linalg.svd(centered.cuda(), full_matrices=False); modes = vh[:32].cpu()
    train_top = centered @ modes[:16].T; train_bulk = centered @ modes[16:32].T
    top_selection = select_individual(train_top.numpy(), train_labels, names, config["minimum_positive"])
    bulk_selection = select_individual(train_bulk.numpy(), train_labels, names, config["minimum_positive"])
    top_decoder = fit_subspace_decoders(train_top, train_labels, config["minimum_positive"], config["ridge"])
    bulk_decoder = fit_subspace_decoders(train_bulk, train_labels, config["minimum_positive"], config["ridge"])
    generator = torch.Generator().manual_seed(config["seed"]); rotations = []
    for _ in range(config["rotation_replicates"]):
        q, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
        rotated_train = train_top @ q
        rotations.append({"q": q, "selection": select_individual(rotated_train.numpy(), train_labels, names, config["minimum_positive"])})
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); matrix, current, next_ids, position = collect(model, spec, config, axes)
        labels, _ = make_labels(current, next_ids, position, vocabulary, target_ids, config)
        top = (matrix - mean) @ modes[:16].T; bulk = (matrix - mean) @ modes[16:32].T
        true = evaluate_individual(top.numpy(), labels, top_selection)
        bulk_result = evaluate_individual(bulk.numpy(), labels, bulk_selection)
        random_medians = [evaluate_individual((top @ row["q"]).numpy(), labels, row["selection"])["median_auc"] for row in rotations]
        results[corpus] = {"top_individual": true, "bulk_individual": bulk_result,
                           "random_rotation_median_auc": random_medians,
                           "random_rotation_p95": float(np.quantile(random_medians, .95)),
                           "top_subspace_decoder": evaluate_subspace(top, labels, top_decoder),
                           "bulk_subspace_decoder": evaluate_subspace(bulk, labels, bulk_decoder)}
    e0 = results["fineweb"]["top_individual"]["median_auc"] >= .75
    e1 = all(results[c]["top_individual"]["median_auc"] >= .70 for c in ("wikitext", "tinystories"))
    e2 = all(row["top_individual"]["median_auc"] >= row["random_rotation_p95"] + .05 for row in results.values())
    e3 = all(row["top_individual"]["median_auc"] >= row["bulk_individual"]["median_auc"] + .05 for row in results.values())
    chosen_non_position = {row["label"] for row in top_selection if not row["label"].startswith("position:")}
    e4 = len(chosen_non_position) >= 5
    e5 = all(row["top_subspace_decoder"]["median_auc"] >= .75 for row in results.values())
    position_count = sum(row["label"].startswith("position:") for row in top_selection); e6 = position_count < 8
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "E0_outlier_modes_have_simple_semantics": e0, "E1_individual_semantics_transfer_ood": e1,
        "E2_pca_axes_are_semantically_privileged": e2, "E3_outlier_modes_are_more_semantic_than_bulk": e3,
        "E4_mode_labels_are_diverse": e4, "E5_mode_subspace_contains_semantic_information": e5,
        "E6_position_does_not_explain_the_modes": e6}.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "label_names": names,
                                  "top_mode_selections": top_selection, "bulk_mode_selections": bulk_selection,
                                  "position_selection_count": position_count, "results": results}, indent=2, allow_nan=False))
    lines = ["# Are spectral Jacobian modes semantic verbs?", "",
             "The top 16 FineWeb transport-Jacobian PCA modes were frozen. Each mode selected one simple current-token, next-token, or position predicate and polarity using only discovery data. Those exact explanations were evaluated on disjoint FineWeb, WikiText, and TinyStories. Twenty random rotations received the identical label-search budget. Modes 17–32 are a spectral-bulk control.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Top-mode median / >=.80 | Bulk median | Random-rotation p95 | Top-subspace decoder |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['top_individual']['median_auc']:.3f} / {row['top_individual']['modes_above_080']} | "
                     f"{row['bulk_individual']['median_auc']:.3f} | {row['random_rotation_p95']:.3f} | "
                     f"{row['top_subspace_decoder']['median_auc']:.3f} |")
    lines += ["", "## Frozen top-mode explanations", "", "| Mode | Predicate | Discovery | FineWeb | WikiText | TinyStories |", "|---:|---|---:|---:|---:|---:|"]
    def display(value): return "n/a" if value is None else f"{value:.3f}"
    for index, selected in enumerate(top_selection):
        lines.append(f"| {index + 1} | `{selected['label']}` | {selected['discovery_auc']:.3f} | "
                     f"{display(results['fineweb']['top_individual']['mode_aucs'][index])} | "
                     f"{display(results['wikitext']['top_individual']['mode_aucs'][index])} | "
                     f"{display(results['tinystories']['top_individual']['mode_aucs'][index])} |")
    lines += ["", "Individual-mode success would support a basis-dependent verb vocabulary. Subspace-only success with random rotations matching PCA would instead mean semantic information is present but mode coordinates are conventional. Failure of both would leave the spectral outliers as reusable causal geometry without a simple token-local semantic description."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
