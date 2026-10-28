#!/usr/bin/env python3
"""Build a large-corpus semantic atlas of compact router axes and edges."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, layer_input, load_snapshot_model
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_router_joint_diagonalization import train_rotation
from run_bilin12_router_pair_graph_operations import coordinates, pair_scores
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_interaction_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_interaction_atlas.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_INTERACTION_ATLAS.md"


def effective_support(ids):
    counts = np.asarray(list(Counter(map(int, ids)).values()), dtype=np.float64)
    return float(counts.sum() ** 2 / np.square(counts).sum())


def fast_auc(labels, scores):
    """Vectorized Mann-Whitney AUROC with average ranks for exact ties."""
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum()); negatives = len(labels) - positives
    if not positives or not negatives: return float("nan")
    order = np.argsort(scores, kind="mergesort"); sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    stops = np.r_[starts[1:], len(scores)]
    average_ranks = (starts + stops + 1) / 2
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.repeat(average_ranks, stops - starts)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def escaped(text):
    return text.encode("unicode_escape").decode("ascii")


def context(rows, row, position, tokenizer, radius=6):
    lo, hi = max(0, position - radius), min(rows.shape[1], position + radius + 1)
    before = tokenizer.decode(rows[row, lo:position].tolist())
    current = tokenizer.decode([int(rows[row, position])])
    after = tokenizer.decode(rows[row, position + 1:hi].tolist())
    return escaped(before + "[" + current + "]" + after)


def best_label(discovery_scores, validation_scores, discovery_labels, validation_labels, counts):
    candidates = []
    for index, name in enumerate(counts):
        if discovery_labels[:, index].sum() < 30 or validation_labels[:, index].sum() < 30:
            continue
        value = fast_auc(discovery_labels[:, index], discovery_scores)
        if not np.isfinite(value): continue
        polarity = 1 if value >= .5 else -1
        candidates.append((max(value, 1 - value), index, polarity, name))
    if not candidates:
        return {"label": None, "discovery_auc": None, "validation_auc": None, "polarity": 0}
    discovery_auc, index, polarity, name = max(candidates)
    validation_auc = fast_auc(validation_labels[:, index], polarity * validation_scores)
    return {"label": name, "discovery_auc": float(discovery_auc),
            "validation_auc": float(validation_auc), "polarity": polarity}


def tails(values, rows, row_ids, positions, current_ids, tokenizer, count):
    order = np.argsort(values)
    def examples(indices):
        return [{"value": float(values[index]), "row": int(row_ids[index]),
                 "position": int(positions[index]), "token_id": int(current_ids[index]),
                 "token": escaped(tokenizer.decode([int(current_ids[index])])),
                 "context": context(rows, int(row_ids[index]), int(positions[index]), tokenizer)}
                for index in indices]
    return {"negative": examples(order[:count]), "positive": examples(order[-count:][::-1])}


def top_tokens(values, ids, tokenizer, minimum=20, count=8):
    size = int(ids.max()) + 1
    counts = np.bincount(ids, minlength=size)
    sums = np.bincount(ids, weights=values, minlength=size)
    valid = np.flatnonzero(counts >= minimum)
    rows = [(float(sums[token] / counts[token]), int(token), int(counts[token])) for token in valid]
    rows.sort()
    convert = lambda part: [{"token_id": int(token), "token": escaped(tokenizer.decode([int(token)])),
                             "mean": mean, "count": int(n)} for mean, token, n in part]
    return {"lowest": convert(rows[:count]), "highest": convert(rows[-count:][::-1])}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    targets = {name: token_id(tokenizer, name) for name in config["targets"]}
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural_states, _ = collect_states_writes(model, 11, config["natural"],
                                               config["natural_positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural_states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms,
                              natural_basis).detach()
    rotation, training = train_rotation(restricted, config); rotation = rotation.detach()
    forms = rotation.T[None] @ restricted @ rotation[None]
    axes = natural_basis @ rotation
    with torch.inference_mode():
        operation_scores = {}
        for name, target in targets.items():
            panel = paired_panel(model, config["graph_discovery"], 11, target, config["batch_size"])
            base, edit = coordinates(model, panel, natural_basis, rotation)
            operation_scores[name] = pair_scores(forms, base, edit)
        pairs = list(operation_scores[next(iter(operation_scores))])
        pooled = {pair: float(np.mean([operation_scores[name][pair] for name in targets]))
                  for pair in pairs}
        selected_pairs = sorted(pairs, key=pooled.get, reverse=True)[:config["pair_budget"]]
        rng = np.random.default_rng(config["seed"]); random_pairs = list(pairs); rng.shuffle(random_pairs)
        random_pairs = random_pairs[:config["pair_budget"]]
        random_rotation = torch.linalg.qr(torch.randn(config["input_rank"], config["input_rank"],
                                                       generator=torch.Generator(device="cuda").manual_seed(config["seed"]),
                                                       device="cuda")).Q
        random_axes = natural_basis @ random_rotation
        source = np.load(config["atlas"]["path"], mmap_mode="r")
        rows = np.asarray(source[config["atlas"]["start"]:config["atlas"]["start"] + config["atlas"]["rows"]],
                          dtype=np.int64)
        positions = np.asarray(config["atlas_positions"], dtype=np.int64)
        axis_values = []; random_axis_values = []
        for start in range(0, len(rows), config["batch_size"]):
            ids = torch.as_tensor(rows[start:start + config["batch_size"], :128], device="cuda")
            states = layer_input(model, ids, 11)[:, positions].float()
            axis_values.append((states @ axes).flatten(0, 1).cpu())
            random_axis_values.append((states @ random_axes).flatten(0, 1).cpu())
            if start and start % 1000 == 0: print("atlas rows", start, flush=True)
        axis_values = torch.cat(axis_values).numpy(); random_axis_values = torch.cat(random_axis_values).numpy()
    pair_values = np.stack([2 * axis_values[:, i] * axis_values[:, j] for i,j in selected_pairs], -1)
    random_pair_values = np.stack([2 * axis_values[:, i] * axis_values[:, j] for i,j in random_pairs], -1)
    row_ids = np.repeat(np.arange(len(rows)), len(positions)); site_positions = np.tile(positions, len(rows))
    current_ids = rows[row_ids, site_positions]; next_ids = rows[row_ids, site_positions + 1]
    vocab_flags = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    labels = np.concatenate((vocab_flags[current_ids], vocab_flags[next_ids]), -1)
    label_names = [f"current:{name}" for name in CATEGORIES] + [f"next:{name}" for name in CATEGORIES]
    split = row_ids < len(rows) // 2

    def audit_matrix(values, kind, items):
        audited = []
        for index in range(values.shape[1]):
            semantic = best_label(values[split, index], values[~split, index], labels[split], labels[~split], label_names)
            top = np.argsort(np.abs(values[:, index]))[-100:]
            row = {"index": index, "semantic": semantic,
                   "top100_current_token_support": effective_support(current_ids[top]),
                   "top_current_tokens": top_tokens(values[:, index], current_ids, tokenizer),
                   "top_next_tokens": top_tokens(values[:, index], next_ids, tokenizer)}
            if kind != "control":
                row["tails"] = tails(values[:, index], rows, row_ids, site_positions,
                                     current_ids, tokenizer, config["tail_examples"])
            if items is not None: row["coordinates"] = list(map(int, items[index])) if isinstance(items[index], tuple) else int(items[index])
            audited.append(row)
        return audited

    axis_rows = audit_matrix(axis_values, "axis", list(range(config["input_rank"])))
    random_axis_rows = audit_matrix(random_axis_values, "control", list(range(config["input_rank"])))
    pair_rows = audit_matrix(pair_values, "pair", selected_pairs)
    random_pair_rows = audit_matrix(random_pair_values, "control", random_pairs)
    degree = Counter(index for pair in selected_pairs for index in pair)
    with torch.inference_mode():
        for row, (i,j) in zip(pair_rows, selected_pairs):
            direction = 2 * forms[:, i, j] @ write_basis.T
            logits = direction @ model.lm_head.weight.float().T
            high = logits.topk(8).indices.tolist(); low = logits.topk(8, largest=False).indices.tolist()
            row["pooled_score"] = pooled[(i,j)]; row["output_direction_norm"] = float(direction.norm())
            row["raw_output_promotes"] = [escaped(tokenizer.decode([index])) for index in high]
            row["raw_output_suppresses"] = [escaped(tokenizer.decode([index])) for index in low]
    axis_validation = [row["semantic"]["validation_auc"] for row in axis_rows if row["semantic"]["validation_auc"] is not None]
    random_axis_validation = [row["semantic"]["validation_auc"] for row in random_axis_rows if row["semantic"]["validation_auc"] is not None]
    pair_validation = [row["semantic"]["validation_auc"] for row in pair_rows if row["semantic"]["validation_auc"] is not None]
    random_pair_validation = [row["semantic"]["validation_auc"] for row in random_pair_rows if row["semantic"]["validation_auc"] is not None]
    supports = [row["top100_current_token_support"] for row in axis_rows + pair_rows]
    hypotheses = {
        "I0_large_panel_completed": {"supported": len(axis_values) >= 90000, "sites": len(axis_values)},
        "I1_canonical_axes_have_simple_token_semantics": {"supported": sum(v >= .80 for v in axis_validation) >= 8,
                                                            "count": sum(v >= .80 for v in axis_validation)},
        "I2_canonical_axes_beat_random_rotations": {"supported": np.median(axis_validation) >= np.median(random_axis_validation) + .05,
                                                     "canonical_median": float(np.median(axis_validation)),
                                                     "random_median": float(np.median(random_axis_validation))},
        "I3_selected_edges_have_simple_token_semantics": {"supported": sum(v >= .80 for v in pair_validation) >= 8,
                                                            "count": sum(v >= .80 for v in pair_validation)},
        "I4_selected_edges_beat_random_edges_semantically": {"supported": np.median(pair_validation) >= np.median(random_pair_validation) + .05,
                                                               "selected_median": float(np.median(pair_validation)),
                                                               "random_median": float(np.median(random_pair_validation))},
        "I5_tail_examples_are_not_single_token_detectors": {"supported": np.median(supports) >= 5,
                                                              "median_support": float(np.median(supports))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "training": training, "hypotheses": hypotheses,
               "selected_pairs": [list(pair) for pair in selected_pairs],
               "axis_degrees": dict(degree), "axes": axis_rows, "pairs": pair_rows,
               "controls": {"random_axes": random_axis_rows, "random_pairs": random_pair_rows}}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Semantic atlas of the compact router graph", "",
             f"The atlas audits {len(axis_values):,} held-out FineWeb token sites. Thirty-two weight-canonicalized coordinates and the pooled 32-edge graph are compared with random rotations/edges. Candidate labels are selected on the first half and evaluated on the second; tail contexts are descriptive held-out examples.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Most label-readable coordinates", "",
              "| Axis | Degree | Held-out label / AUROC | Top-100 token support | Positive example | Negative example |",
              "|---:|---:|---|---:|---|---|"]
    for row in sorted(axis_rows, key=lambda r: r["semantic"]["validation_auc"] or -1, reverse=True)[:12]:
        positive = row["tails"]["positive"][0]["context"]; negative = row["tails"]["negative"][0]["context"]
        lines.append(f"| {row['index']} | {degree.get(row['index'],0)} | {row['semantic']['label']} / {row['semantic']['validation_auc']:.3f} | "
                     f"{row['top100_current_token_support']:.1f} | `{positive}` | `{negative}` |")
    lines += ["", "## Most label-readable selected interactions", "",
              "| Edge | Held-out label / AUROC | Support | Raw promoted tokens | Positive example |",
              "|---|---|---:|---|---|"]
    for row in sorted(pair_rows, key=lambda r: r["semantic"]["validation_auc"] or -1, reverse=True)[:16]:
        edge = "×".join(map(str, row["coordinates"])); promoted = ", ".join(f"`{token}`" for token in row["raw_output_promotes"][:5])
        lines.append(f"| {edge} | {row['semantic']['label']} / {row['semantic']['validation_auc']:.3f} | "
                     f"{row['top100_current_token_support']:.1f} | {promoted} | `{row['tails']['positive'][0]['context']}` |")
    lines += ["", "A high category AUROC gives a simple descriptive handle, not a complete meaning. A coordinate can participate in several edges with different output directions; an edge can be a clean relational verb even if neither endpoint is independently monosemantic. Raw output tokens use the unembedding without the context-dependent final RMSNorm and are therefore role hints, not exact causal effects."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
