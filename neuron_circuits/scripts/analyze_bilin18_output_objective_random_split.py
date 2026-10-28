#!/usr/bin/env python3
"""Random-split robustness check for label-free output-axis discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_router_jacobian_sparse_basis_redteam import greedy_overlap, varimax
from run_bilin18_layer17_tangent_output_semantics import evaluate_individual, select_individual, vocabulary_table
from run_bilin18_output_objective_sparse_discovery import absolute_excess_kurtosis, standardized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "bilin18_output_objective_sparse_discovery.json"
OUTPUT = TENSOR_ROOT / "bilin18_output_objective_random_split.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_OBJECTIVE_RANDOM_SPLIT.md"


def main():
    source = json.loads(SOURCE.read_text()); config = source["config"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    _, labels, names = vocabulary_table(tokenizer)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda"); model.lm_head.float().eval()
    scores = (model.lm_head.weight.float()[:tokenizer.vocab_size] @ axes).detach().cpu()
    generator = np.random.default_rng(config["seed"] + 91)
    torch_generator = torch.Generator().manual_seed(config["seed"] + 91)
    rows, rotations = [], []
    for split in range(8):
        train = generator.random(tokenizer.vocab_size) < .5; test = ~train
        z, _, _ = standardized(scores[train]); q, metadata = varimax(z); rotations.append(q)
        selection = select_individual((scores[train] @ q).numpy(), labels[train], names, config["minimum_positive"])
        semantic = evaluate_individual((scores[test] @ q).numpy(), labels[test], selection)["median_auc"]
        kurtosis = absolute_excess_kurtosis(scores[test] @ q)
        null_semantic, null_kurtosis = [], []
        for _ in range(16):
            random_q = torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=torch_generator)).Q
            random_selection = select_individual((scores[train] @ random_q).numpy(), labels[train], names, config["minimum_positive"])
            null_semantic.append(evaluate_individual((scores[test] @ random_q).numpy(), labels[test], random_selection)["median_auc"])
            null_kurtosis.append(absolute_excess_kurtosis(scores[test] @ random_q))
        rows.append({"split": split, "semantic_auc": float(semantic), "semantic_random_p95": float(np.quantile(null_semantic, .95)), "kurtosis": float(kurtosis), "kurtosis_random_p95": float(np.quantile(null_kurtosis, .95)), "varimax_iterations": metadata["iterations"]})
    overlaps = [greedy_overlap(rotations[left], rotations[right]) for left in range(len(rotations)) for right in range(left + 1, len(rotations))]
    summary = {"median_pairwise_axis_overlap": float(np.median(overlaps)), "minimum_pairwise_axis_overlap": float(np.min(overlaps)), "median_semantic_auc": float(np.median([row["semantic_auc"] for row in rows])), "median_semantic_minus_random_p95": float(np.median([row["semantic_auc"] - row["semantic_random_p95"] for row in rows])), "median_kurtosis": float(np.median([row["kurtosis"] for row in rows])), "median_kurtosis_minus_random_p95": float(np.median([row["kurtosis"] - row["kurtosis_random_p95"] for row in rows]))}
    OUTPUT.write_text(json.dumps({"source": str(SOURCE), "status": "posthoc-random-split-redteam", "summary": summary, "pairwise_overlaps": overlaps, "splits": rows}, indent=2))
    REPORT.write_text("\n".join(["# Random-split red team of the sparse output basis", "", "Eight seeded random half-vocabulary splits replace the original even/odd split. Varimax is fitted without labels on each training half. Held-out sparsity and semantic label search are compared with sixteen matched random rotations per split.", "", f"Median/minimum pairwise recovered-axis overlap: **{summary['median_pairwise_axis_overlap']:.3f}/{summary['minimum_pairwise_axis_overlap']:.3f}**.", f"Median held-out kurtosis and advantage over random p95: **{summary['median_kurtosis']:.3f}/{summary['median_kurtosis_minus_random_p95']:+.3f}**.", f"Median held-out semantic AUROC and advantage over random p95: **{summary['median_semantic_auc']:.3f}/{summary['median_semantic_minus_random_p95']:+.3f}**.", "", "The random splits test whether tokenizer-ID parity manufactured the stable sparse coordinates. They remain a post-hoc robustness check, not a new preregistered semantic claim."]) + "\n")
    print(json.dumps(summary, indent=2)); print(REPORT)


if __name__ == "__main__": main()
