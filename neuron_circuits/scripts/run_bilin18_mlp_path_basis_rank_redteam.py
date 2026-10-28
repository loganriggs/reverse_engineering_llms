#!/usr/bin/env python3
"""Red-team late path compactness across natural/random endpoint charts and rank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_symmetry_redteam import GROUPS, collapse
from run_bilin12_mlp_path_type_program import aggregate_gram, greedy_order, path_fidelity, path_matrices
from run_bilin18_mlp_path_symmetry_replication import load_rows, trace_suffix
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_mlp_path_basis_rank_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_mlp_path_basis_rank_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_MLP_PATH_BASIS_RANK_REDTEAM.md"


@torch.inference_mode()
def pca_axes(model, base, maximum_rank):
    inputs = []; outputs = []; source = load_rows(base["axis_discovery"])
    for start in range(0, len(source), base["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + base["batch_size"], :128], dtype=np.int64), device="cuda")
        x, _, final, _ = trace_suffix(model, ids, base)
        inputs.append(x[:, base["positions"]].float().flatten(0, 1).cpu())
        outputs.append(final[:, base["positions"]].float().flatten(0, 1).cpu())
    _, _, input_vh = torch.linalg.svd(torch.cat(inputs), full_matrices=False)
    _, _, output_vh = torch.linalg.svd(torch.cat(outputs), full_matrices=False)
    return input_vh[:maximum_rank].T.cuda(), output_vh[:maximum_rank].T.cuda()


def random_axes(dimension, rank, generator):
    first = torch.linalg.qr(torch.randn(dimension, rank, generator=generator, device="cuda")).Q
    second = torch.linalg.qr(torch.randn(dimension, rank, generator=generator, device="cuda")).Q
    return first, second


@torch.inference_mode()
def collect(model, input_axes, output_axes, spec, base):
    source = load_rows(spec); stores = []
    local = {**base, "rank": input_axes.shape[1]}
    for start in range(0, len(source), base["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + base["batch_size"], :128], dtype=np.int64), device="cuda")
        for position in base["positions"]:
            stores.append(collapse(path_matrices(
                model, ids, position, local, input_axes, output_axes
            ).cpu()))
    return torch.cat(stores)


def k90(paths, ordering):
    for count in range(1, 17):
        if path_fidelity(paths, ordering[:count]) >= .90:
            return count
    return 16


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    maximum_rank = max(config["ranks"]); natural_input, natural_output = pca_axes(model, base, maximum_rank)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    axes = {}
    for rank in config["ranks"]:
        axes[f"pca{rank}"] = (natural_input[:, :rank], natural_output[:, :rank])
        axes[f"random{rank}"] = random_axes(model.config.n_embd, rank, generator)
    results = {}; orders = {}
    for name, (input_axes, output_axes) in axes.items():
        print("discover", name, flush=True)
        discovery = collect(model, input_axes, output_axes, base["path_discovery"], base)
        ordering = greedy_order(*aggregate_gram(discovery)); orders[name] = ordering
        corpus_rows = {}
        for corpus, spec in base["evaluation"].items():
            print("evaluate", name, corpus, flush=True)
            paths = collect(model, input_axes, output_axes, spec, base)
            corpus_rows[corpus] = {"k4_fidelity": path_fidelity(paths, ordering[:4]),
                                   "k8_fidelity": path_fidelity(paths, ordering[:8]),
                                   "k90": k90(paths, ordering)}
        results[name] = corpus_rows

    reference = set(orders["pca32"][:4])
    overlap = {name: len(reference & set(order[:4])) for name, order in orders.items()}
    supported = {
        "B0_pca_rank64_remains_compact": all(row["k4_fidelity"] >= .85 for row in results["pca64"].values()),
        "B1_random_rank32_remains_compact": all(row["k4_fidelity"] >= .85 for row in results["random32"].values()),
        "B2_random_rank64_remains_compact": all(row["k4_fidelity"] >= .85 for row in results["random64"].values()),
        "B3_every_chart_is_low_order": all(row["k90"] <= 5 for chart in results.values() for row in chart.values()),
        "B4_group_order_is_chart_stable": all(value >= 3 for name, value in overlap.items() if name != "pca32"),
        "B5_natural_chart_is_privileged": all(results[f"pca{rank}"][corpus]["k4_fidelity"] >= results[f"random{rank}"][corpus]["k4_fidelity"] + .03 for rank in config["ranks"] for corpus in base["evaluation"]),
        "B6_endpoint_rank_is_stable": all(abs(results["pca64"][corpus]["k4_fidelity"] - results["pca32"][corpus]["k4_fidelity"]) <= .03 for corpus in base["evaluation"]),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    payload = {"config": config, "hypotheses": hypotheses,
               "orders": {name: ["".join(GROUPS[i]) for i in order] for name, order in orders.items()},
               "top4_overlap_with_pca32": overlap, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Endpoint-basis and rank red-team in the 546M model", "",
             "The exact 16-path suffix program is repeated in natural PCA and Haar-random input/output charts at ranks 32 and 64. Each chart gets its own FineWeb discovery ordering; all evaluation remains disjoint and OOD corpora are untouched during selection.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Chart | Top-4 overlap | FineWeb K4/K8/K90 | WikiText | TinyStories |",
              "|---|---:|---:|---:|---:|"]
    for name, chart in results.items():
        cells = []
        for corpus in ("fineweb", "wikitext", "tinystories"):
            row = chart[corpus]; cells.append(f"{row['k4_fidelity']:.3f}/{row['k8_fidelity']:.3f}/{row['k90']}")
        lines.append(f"| {name} | {overlap[name]}/4 | " + " | ".join(cells) + " |")
    lines += ["", "If random charts are equally compact, low interaction order is a property of the local residual operator chain rather than a semantic endpoint basis. If only PCA charts are compact, natural activity selects a privileged contractive slice. Rank stability tests whether the 32D endpoint hid broad off-chart propagation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
