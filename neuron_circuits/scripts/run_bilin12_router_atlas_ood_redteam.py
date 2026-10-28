#!/usr/bin/env python3
"""Evaluate FineWeb-selected compact-router labels on two OOD corpora."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, layer_input, load_snapshot_model
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_router_interaction_atlas import fast_auc
from run_bilin12_router_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_atlas_ood_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_atlas_ood_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_ATLAS_OOD_REDTEAM.md"


def label_index(name):
    side, category = name.split(":", 1)
    return (0 if side == "current" else len(CATEGORIES)) + CATEGORIES.index(category)


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); parent_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(config["parent_result"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(parent_config["snapshot"]), "cuda")
    # Reconstruct the frozen coordinate system; the output basis is only needed
    # to define the same joint forms and rotation used by the parent atlas.
    write_basis = pooled_basis(model, tokenizer, {**parent_config, "rank": parent_config["output_rank"],
                                                   "discovery": parent_config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(),
                       channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, parent_config["natural"],
                                       parent_config["natural_positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False)
    natural_basis = vh[:parent_config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, parent_config); rotation = rotation.detach()
    axes = natural_basis @ rotation
    generator = torch.Generator(device="cuda").manual_seed(parent_config["seed"])
    random_rotation = torch.linalg.qr(torch.randn(parent_config["input_rank"], parent_config["input_rank"],
                                                   generator=generator, device="cuda")).Q
    random_axes = natural_basis @ random_rotation
    selected_pairs = [tuple(row["coordinates"]) for row in parent["pairs"]]
    random_pairs = [tuple(row["coordinates"]) for row in parent["controls"]["random_pairs"]]
    vocab_flags = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positions = np.asarray(parent_config["atlas_positions"], dtype=np.int64)
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        source = np.load(spec["path"], mmap_mode="r")
        rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]], dtype=np.int64)
        values = []; random_values = []
        with torch.inference_mode():
            for start in range(0, len(rows), config["batch_size"]):
                ids = torch.as_tensor(rows[start:start + config["batch_size"], :128], device="cuda")
                state = layer_input(model, ids, 11)[:, positions].float()
                values.append((state @ axes).flatten(0, 1).cpu())
                random_values.append((state @ random_axes).flatten(0, 1).cpu())
        values = torch.cat(values).numpy(); random_values = torch.cat(random_values).numpy()
        pair_values = np.stack([2 * values[:, i] * values[:, j] for i,j in selected_pairs], -1)
        random_pair_values = np.stack([2 * values[:, i] * values[:, j] for i,j in random_pairs], -1)
        row_ids = np.repeat(np.arange(len(rows)), len(positions)); sites = np.tile(positions, len(rows))
        labels = np.concatenate((vocab_flags[rows[row_ids, sites]], vocab_flags[rows[row_ids, sites + 1]]), -1)
        def score_rows(matrix, interpretations):
            output = []
            for index, interpretation in enumerate(interpretations):
                semantic = interpretation["semantic"]; name = semantic["label"]
                if name is None: output.append(None); continue
                value = fast_auc(labels[:, label_index(name)], semantic["polarity"] * matrix[:, index])
                output.append(float(value) if np.isfinite(value) else None)
            return output
        axis_auc = score_rows(values, parent["axes"])
        random_axis_auc = score_rows(random_values, parent["controls"]["random_axes"])
        edge_auc = score_rows(pair_values, parent["pairs"])
        random_edge_auc = score_rows(random_pair_values, parent["controls"]["random_pairs"])
        finite_axis = [value for value in axis_auc if value is not None]
        finite_random_axis = [value for value in random_axis_auc if value is not None]
        finite_edge = [value for value in edge_auc if value is not None]
        finite_random_edge = [value for value in random_edge_auc if value is not None]
        results[corpus] = {"sites": len(values), "axis_auc": axis_auc,
                           "random_axis_auc": random_axis_auc, "edge_auc": edge_auc,
                           "random_edge_auc": random_edge_auc,
                           "axis_median": float(np.median(finite_axis)),
                           "random_axis_median": float(np.median(finite_random_axis)),
                           "edge_median": float(np.median(finite_edge)),
                           "random_edge_median": float(np.median(finite_random_edge)),
                           "eligible_edges": len(finite_edge),
                           "edges_at_080": int(sum(value >= .80 for value in finite_edge))}
    s0 = all(row["edge_median"] >= .85 for row in results.values())
    count_both = sum(all(results[c]["edge_auc"][i] is not None
                         and results[c]["edge_auc"][i] >= .80 for c in results) for i in range(32))
    s1 = count_both >= 24
    s2 = all(row["edge_median"] >= row["random_edge_median"] + .10 for row in results.values())
    s3 = all(row["axis_median"] >= .70 for row in results.values())
    labels_used = {row["semantic"]["label"] for row in parent["pairs"] if row["semantic"]["label"]}
    s4 = len(labels_used) >= 4
    hypotheses = {
        "S0_edge_semantics_transfer_ood": {"supported": s0},
        "S1_most_edges_transfer_ood": {"supported": s1,
                                         "count_both": count_both},
        "S2_selected_edges_beat_random_edges_ood": {"supported": s2},
        "S3_axis_semantics_transfer_ood": {"supported": s3},
        "S4_semantic_labels_are_not_all_one_predicate": {"supported": s4,
                                                            "labels": sorted(labels_used)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2, allow_nan=False))
    lines = ["# OOD red team of compact-router semantics", "",
             "Every axis/edge label and polarity is frozen from the FineWeb atlas, then evaluated without reselection on all sites from WikiText-103 and TinyStories. Random rotations and random edges retain their own FineWeb-selected labels as matched controls.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Sites | Axis/random median AUROC | Edge/random median AUROC | Edges >=0.80 |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['sites']} | {row['axis_median']:.3f}/{row['random_axis_median']:.3f} | "
                     f"{row['edge_median']:.3f}/{row['random_edge_median']:.3f} | {row['edges_at_080']}/{row['eligible_edges']} |")
    lines += ["", "OOD transfer supports a stable descriptive predicate, not a complete mechanistic meaning. If many edges share the same label, the result is one distributed boundary code with multiple output interactions—not 32 independent semantic relations."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
