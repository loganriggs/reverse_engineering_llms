#!/usr/bin/env python3
"""Do stable residual-Fisher objectives reuse fixed early native channels?"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_output_basis_upstream_compactness import propagate_source_adjoints
from run_bilin18_output_metric_whitening_redteam import residual_adjoints
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_residual_fisher_channel_reuse.json")
OUTPUT = TENSOR_ROOT / "bilin18_residual_fisher_channel_reuse.json"
REPORT = TENSOR_ROOT / "BILIN18_RESIDUAL_FISHER_CHANNEL_REUSE.md"
FIGURE = TENSOR_ROOT / "bilin18_residual_fisher_channel_reuse.png"


def collect(model, rows, config, axes):
    sites = []
    for row_index, raw_row in enumerate(rows):
        ids = torch.as_tensor(np.asarray(raw_row[:128], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in config["positions"]:
            final = residual_adjoints(trace[-1]["output"], position, axes)
            gradients = propagate_source_adjoints(model, trace, position, final, config["source_layers"])
            for layer in config["source_layers"]:
                sites.append({"row": row_index, "position": position, "layer": layer, "gradient": gradients[layer].detach().cpu(), "product": product_at(model, trace, layer, position).detach().cpu()})
        print(f"Fisher channel rows {row_index + 1}/{len(rows)}", flush=True)
    return sites


def fit_supports(sites, bases, layers, rank):
    supports = {name: {} for name in bases}
    for name, q in bases.items():
        for layer in layers:
            rows = [site for site in sites if site["layer"] == layer]
            gradient = torch.stack([q.T @ site["gradient"] for site in rows])
            product = torch.stack([site["product"] for site in rows])
            contribution = gradient * product[:, None]
            supports[name][layer] = {"gradient": gradient.square().mean(0).topk(rank, dim=-1).indices, "contribution": contribution.square().mean(0).topk(rank, dim=-1).indices}
    return supports


def fraction(values, support, squared):
    mass = values.abs().square() if squared else values.abs()
    selected = torch.gather(mass, -1, support[None].expand(len(mass), -1))
    return selected.sum(-1) / mass.sum(-1).clamp_min(1e-30)


def jaccard(first, second):
    a, b = set(first.tolist()), set(second.tolist()); return len(a & b) / len(a | b)


def evaluate(sites, bases, supports, layers, rank):
    records = []
    for name, q in bases.items():
        for layer in layers:
            rows = [site for site in sites if site["layer"] == layer]
            gradients = torch.stack([q.T @ site["gradient"] for site in rows]); products = torch.stack([site["product"] for site in rows]); contributions = gradients * products[:, None]
            for family, values, squared in (("gradient", gradients, True), ("contribution", contributions, False)):
                fixed = supports[name][layer][family]
                fixed_values = torch.stack([fraction(values[:, axis], fixed[axis], squared) for axis in range(values.shape[1])], 1)
                oracle_values, endpoint_supports = [], []
                for site_index in range(values.shape[0]):
                    local = values[site_index].abs().square() if squared else values[site_index].abs()
                    local_support = local.topk(rank, dim=-1).indices; endpoint_supports.append(local_support)
                    oracle_values.append(torch.gather(local, -1, local_support).sum(-1) / local.sum(-1).clamp_min(1e-30))
                oracle_values = torch.stack(oracle_values); endpoint_supports = torch.stack(endpoint_supports)
                same = [jaccard(endpoint_supports[index, axis], endpoint_supports[index + 1, axis]) for index in range(len(rows) - 1) for axis in range(values.shape[1])]
                different = [jaccard(endpoint_supports[index, axis], endpoint_supports[index, (axis + 1) % values.shape[1]]) for index in range(len(rows)) for axis in range(values.shape[1])]
                records.append({"basis": name, "layer": layer, "family": family, "fixed_capture": float(fixed_values.median()), "oracle_capture": float(oracle_values.median()), "fixed_to_oracle": float(fixed_values.median() / oracle_values.median().clamp_min(1e-30)), "leading4_fixed_capture": float(fixed_values[:, :4].median()), "trailing12_fixed_capture": float(fixed_values[:, 4:].median()), "same_axis_support_jaccard": float(np.median(same)), "different_axis_support_jaccard": float(np.median(different))})
    return records


def row(records, basis, layer, family):
    return next(item for item in records if item["basis"] == basis and item["layer"] == layer and item["family"] == family)


def main():
    config = json.loads(CONFIG.read_text()); core = torch.load(config["core_path"], map_location="cpu", weights_only=False); axes = core["output"][:, :config["rank"]].float().cuda(); saved = torch.load(config["fisher_axes_path"], map_location="cpu", weights_only=False)
    fisher_q = saved["discovery"]["residual_dual"].float(); generator = torch.Generator().manual_seed(config["seed"]); random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(config["random_rotations"])]
    bases = {"residual_fisher": fisher_q, **{f"random_{index}": q for index, q in enumerate(random_qs)}}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r"); discovery_rows = corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"]]; evaluation_rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"]]
    print("collect reuse discovery", flush=True); discovery = collect(model, discovery_rows, config, axes)
    supports = fit_supports(discovery, bases, config["source_layers"], config["support_rank"])
    print("collect reuse evaluation", flush=True); evaluation = collect(model, evaluation_rows, config, axes)
    records = evaluate(evaluation, bases, supports, config["source_layers"], config["support_rank"])
    random_p95 = {}
    for layer in config["source_layers"]:
        random_p95[str(layer)] = {}
        for family in ("gradient", "contribution"):
            random_p95[str(layer)][family] = {metric: float(np.quantile([row(records, f"random_{index}", layer, family)[metric] for index in range(config["random_rotations"])], .95)) for metric in ("fixed_capture", "fixed_to_oracle", "same_axis_support_jaccard")}
    hypotheses = {
        "R0_fisher_axes_have_privileged_frozen_gradient_support": {"supported": all(row(records, "residual_fisher", layer, "gradient")["fixed_capture"] >= random_p95[str(layer)]["gradient"]["fixed_capture"] + .02 for layer in config["source_layers"])},
        "R1_fisher_axes_have_privileged_frozen_contribution_support": {"supported": all(row(records, "residual_fisher", layer, "contribution")["fixed_capture"] >= random_p95[str(layer)]["contribution"]["fixed_capture"] + .02 for layer in config["source_layers"])},
        "R2_frozen_support_approaches_endpoint_oracle": {"supported": all(row(records, "residual_fisher", layer, family)["fixed_to_oracle"] >= .50 for layer in config["source_layers"] for family in ("gradient", "contribution"))},
        "R3_same_axis_support_recurs": {"supported": all(row(records, "residual_fisher", layer, family)["same_axis_support_jaccard"] >= row(records, "residual_fisher", layer, family)["different_axis_support_jaccard"] + .02 for layer in config["source_layers"] for family in ("gradient", "contribution"))},
        "R4_leading_axes_are_more_reusable": {"supported": all(row(records, "residual_fisher", layer, "gradient")["leading4_fixed_capture"] >= row(records, "residual_fisher", layer, "gradient")["trailing12_fixed_capture"] + .02 for layer in config["source_layers"])},
    }
    output = {"config": config, "hypotheses": hypotheses, "residual_fisher": {str(layer): {family: row(records, "residual_fisher", layer, family) for family in ("gradient", "contribution")} for layer in config["source_layers"]}, "random_p95": random_p95, "records": records}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    lines = ["# Do stable residual-Fisher axes reuse fixed native channels?", "", "The residual-dual Fisher basis is frozen from the prior metric experiment. Discovery FineWeb ranks native channels separately for each axis using mean squared gradient or mean squared native activation-weighted contribution. Disjoint FineWeb compares those fixed top-64 lists with each endpoint's own top 64 and with 32 random rotations of the identical output subspace.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Fisher channel reuse](bilin18_residual_fisher_channel_reuse.png)", "", "| Layer | Family | Frozen capture | Endpoint oracle | Frozen/oracle | Same/different Jaccard | Random frozen p95 | Leading4/trailing12 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for family in ("gradient", "contribution"):
            value = row(records, "residual_fisher", layer, family); null = random_p95[str(layer)][family]
            lines.append(f"| {layer} | {family} | {value['fixed_capture']:.3f} | {value['oracle_capture']:.3f} | {value['fixed_to_oracle']:.3f} | {value['same_axis_support_jaccard']:.3f}/{value['different_axis_support_jaccard']:.3f} | {null['fixed_capture']:.3f} | {value['leading4_fixed_capture']:.3f}/{value['trailing12_fixed_capture']:.3f} |")
    lines += ["", "## Guardrails", "", "Fisher axes are stable objective directions under the residual metric. This test asks a different question: whether each objective owns a reusable list of native early channels. A null result does not invalidate the objective subspace; it means stable behavior is executed by context-conditioned coalitions rather than fixed neuron identities."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, plots = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for family in ("gradient", "contribution"):
        plots[0].plot(x, [row(records, "residual_fisher", layer, family)["fixed_capture"] for layer in config["source_layers"]], marker="o", label=f"{family} frozen")
        plots[0].plot(x, [row(records, "residual_fisher", layer, family)["oracle_capture"] for layer in config["source_layers"]], marker="x", linestyle="--", label=f"{family} oracle")
        plots[1].plot(x, [row(records, "residual_fisher", layer, family)["same_axis_support_jaccard"] for layer in config["source_layers"]], marker="o", label=f"{family} same axis")
        plots[1].plot(x, [row(records, "residual_fisher", layer, family)["different_axis_support_jaccard"] for layer in config["source_layers"]], marker="x", linestyle="--", label=f"{family} different axis")
    plots[0].set(xlabel="Source layer", ylabel="Captured mass", title="Frozen versus endpoint support"); plots[1].set(xlabel="Source layer", ylabel="Top-64 Jaccard", title="Native support recurrence")
    for plot in plots: plot.set_xticks(x, config["source_layers"]); plot.grid(alpha=.2); plot.legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "residual_fisher": output["residual_fisher"], "random_p95": random_p95}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
