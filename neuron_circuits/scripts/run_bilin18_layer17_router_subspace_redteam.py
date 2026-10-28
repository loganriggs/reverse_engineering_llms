#!/usr/bin/env python3
"""Red-team the causal depth trace with fitted and random state subspaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer17_router_compilation_depth import (
    collect_prediction,
    fit_response,
    gram_fidelity,
    prepare_ridge,
    suffix_state,
    trace_boundaries,
)
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_router_subspace_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_router_subspace_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_ROUTER_SUBSPACE_REDTEAM.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_router_subspace_redteam.png")


def adapted_config(config):
    return {
        **config,
        "prediction_positions": config["positions"],
        "prediction_boundaries": config["boundaries"],
        "prediction_batch_size": config["batch_size"],
    }


def fit_router_subspaces(discovery, config):
    target = discovery["coefficients"]
    routers, randoms, means = {}, {}, {}
    generator = torch.Generator().manual_seed(config["seed"])
    for layer in config["boundaries"]:
        print("fit router", layer, flush=True)
        x = discovery["states"][layer]
        fit = fit_response(prepare_ridge(x, config["ridge_relative_penalty"]), target)
        routers[layer] = torch.linalg.qr(fit["weight"].float()).Q[:, :config["rank"]].cuda()
        randoms[layer] = torch.linalg.qr(torch.randn(
            x.shape[1], config["rank"], generator=generator)).Q.cuda()
        layer_means = []
        for position in config["positions"]:
            layer_means.append(x[discovery["position"] == position].float().mean(0))
        means[layer] = torch.stack(layer_means).cuda()
    return routers, randoms, means


def projected_edit(current, replacement, basis):
    delta = replacement - current
    return current + (delta @ basis) @ basis.T


@torch.inference_mode()
def causal_subspace_trace(model, spec, means, routers, randoms, context_axes, gram, config):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    boundaries = config["boundaries"]
    positions = config["positions"]
    methods = (
        "full_mean", "full_cycle", "router_mean", "router_cycle",
        "random_mean", "random_cycle",
    )
    targets = {layer: [] for layer in boundaries}
    predictions = {layer: {method: [] for method in methods} for layer in boundaries}
    exact_error = 0.0; exact_energy = 0.0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, states = trace_boundaries(model, ids, set(boundaries))
        _, reference = prefix_to_layer17(model, ids)
        native = suffix_state(model, states[16], value_bus, x0, 16)
        reference_selected = torch.cat([reference[:, position] @ context_axes for position in positions])
        native_selected = torch.cat([native[:, position] @ context_axes for position in positions])
        exact_error += float((reference_selected.double() - native_selected.double()).square().sum())
        exact_energy += float(reference_selected.double().square().sum())
        batch = len(ids)
        target = reference_selected.cpu()
        for boundary in boundaries:
            chunks = []
            for method in methods:
                kind, replacement_kind = method.split("_")
                basis = routers[boundary] if kind == "router" else randoms[boundary]
                for index, position in enumerate(positions):
                    edited = states[boundary].clone()
                    current = states[boundary][:, position]
                    replacement = (means[boundary][index].expand_as(current)
                                   if replacement_kind == "mean"
                                   else torch.roll(current, 1, 0))
                    if kind == "full":
                        edited[:, position] = replacement
                    else:
                        edited[:, position] = projected_edit(current, replacement, basis)
                    chunks.append(edited)
            variants = torch.cat(chunks)
            repeats = len(chunks)
            final = suffix_state(
                model, variants,
                value_bus.repeat(*([repeats] + [1] * (value_bus.ndim - 1))),
                x0.repeat(repeats, 1, 1), boundary)
            position_vector = torch.as_tensor(
                [position for _method in methods for position in positions for _ in range(batch)],
                device="cuda")
            selected = final[torch.arange(len(final), device="cuda"), position_vector] @ context_axes
            split = len(positions) * batch
            targets[boundary].append(target)
            for method_index, method in enumerate(methods):
                predictions[boundary][method].append(
                    selected[method_index * split:(method_index + 1) * split].cpu())
    result = {}
    for boundary in boundaries:
        target = torch.cat(targets[boundary])
        result[str(boundary)] = {
            method: gram_fidelity(target, torch.cat(rows), gram)
            for method, rows in predictions[boundary].items()
        }
        result[str(boundary)]["examples"] = len(target)
    return result, float(np.sqrt(exact_error / max(exact_energy, 1e-30)))


def crossing(rows, threshold):
    return next((int(depth) for depth in sorted(rows, key=int)
                 if rows[depth]["full_mean"] < threshold), 18)


def plot_results(results, config):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1), sharey=True)
    for axis, (corpus, rows) in zip(axes, results.items()):
        depths = config["boundaries"]
        for method, label, style in (
            ("full_cycle", "full state", "-o"),
            ("router_cycle", "router span", "-s"),
            ("random_cycle", "random span", "--^"),
        ):
            axis.plot(depths, [rows[str(depth)][method] for depth in depths], style, label=label)
        axis.set_title(corpus)
        axis.set_xlabel("block boundary")
        axis.grid(alpha=.2)
        for threshold in config["thresholds"]:
            axis.axhline(threshold, color="#9ca3af", linestyle=":", linewidth=.8)
    axes[0].set_ylabel("cyclic-patch operator fidelity")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_result"]).read_text())
    trace_config = adapted_config(config)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)
    flat = tensor.reshape(-1, config["rank"]).double()
    gram = flat.T @ flat

    print("collect discovery", flush=True)
    discovery = collect_prediction(model, config["discovery"], trace_config, context_axes)
    routers, randoms, means = fit_router_subspaces(discovery, config)
    results = {}; exact_errors = {}
    for corpus, spec in config["evaluation"].items():
        print("causal", corpus, flush=True)
        results[corpus], exact_errors[corpus] = causal_subspace_trace(
            model, spec, means, routers, randoms, context_axes, gram, config)

    overlap = sorted(set(config["boundaries"]) & set(parent["config"]["causal_boundaries"]))
    replication_differences = {}
    for corpus in results:
        values = []
        for depth in overlap:
            values.extend([
                abs(results[corpus][str(depth)]["full_mean"]
                    - parent["causal"][corpus][str(depth)]["mean_replacement"]),
                abs(results[corpus][str(depth)]["full_cycle"]
                    - parent["causal"][corpus][str(depth)]["cyclic_substitution"]),
            ])
        replication_differences[corpus] = max(values)

    merged = {}
    crossings = {}
    for corpus in results:
        merged[corpus] = {
            str(depth): {
                "full_mean": row["mean_replacement"],
                "full_cycle": row["cyclic_substitution"],
            }
            for depth, row in ((int(depth), row) for depth, row in parent["causal"][corpus].items())
        }
        merged[corpus].update({depth: row for depth, row in results[corpus].items()})
        crossings[corpus] = {
            str(threshold): crossing(merged[corpus], threshold)
            for threshold in config["thresholds"]
        }

    loss_fraction = lambda row, depth: (
        (1 - row[str(depth)]["router_cycle"])
        / max(1e-30, 1 - row[str(depth)]["full_cycle"]))
    hypotheses = {
        "S0_native_suffix_recomposition_is_exact": max(exact_errors.values()) <= 1e-6,
        "S1_full_state_trace_replicates": max(replication_differences.values()) <= .10,
        "S2_router_subspace_is_causally_relevant_at_layer16": all(
            loss_fraction(row, 16) >= .50 for row in results.values()),
        "S3_router_subspace_beats_random_at_layer16": all(
            row["16"]["router_cycle"] <= row["16"]["random_cycle"] - .10
            for row in results.values()),
        "S4_early_predictive_span_is_recoverable": all(
            row["4"]["router_cycle"] >= .90 for row in results.values()),
        "S5_compilation_curve_is_gradual": all(max(
            row[str(depth)]["full_mean"] - row[str(depth + 1)]["full_mean"]
            for depth in range(4, 10)) <= .12 for row in results.values()),
        "S6_single_boundary_is_threshold_dependent": all(
            len(set(row.values())) > 1 for row in crossings.values()),
        "S7_strict_commitment_depth_is_corpus_dependent": (
            max(row["0.8"] for row in crossings.values())
            - min(row["0.8"] for row in crossings.values()) >= 4),
        "S8_router_span_fraction_grows_with_depth": all(
            loss_fraction(row, 16) > loss_fraction(row, 4) for row in results.values()),
        "S9_random_subspace_is_not_equally_causal": all(
            np.mean([1 - row[str(depth)]["router_cycle"] for depth in config["boundaries"]])
            > np.mean([1 - row[str(depth)]["random_cycle"] for depth in config["boundaries"]])
            for row in results.values()),
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "native_recomposition_errors": exact_errors,
        "parent_replication_max_difference": replication_differences,
        "threshold_crossings": crossings,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    plot_results(results, config)

    lines = [
        "# Causal router-subspace and threshold red team", "",
        "The predictive ridge map defines a 16-dimensional input span at each layer. Mean and cyclic edits are restricted to that span, to a matched random span, or to the full state before exact suffix recomputation.",
        "", "![Router subspace red team](bilin18_layer17_router_subspace_redteam.png)",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Cyclic-substitution fidelity", "",
              "| Layer | " + " | ".join(f"{corpus} full/router/random" for corpus in results) + " |",
              "|---:|" + "---:|" * len(results)]
    for depth in config["boundaries"]:
        values = []
        for row in results.values():
            value = row[str(depth)]
            values.append(f"{value['full_cycle']:.3f}/{value['router_cycle']:.3f}/{value['random_cycle']:.3f}")
        lines.append(f"| {depth} | " + " | ".join(values) + " |")
    lines += ["", "## Mean-replacement threshold crossings", "",
              "| Corpus | first below 0.95 | first below 0.90 | first below 0.80 | parent max difference |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in crossings.items():
        lines.append(
            f"| {corpus} | {row['0.95']} | {row['0.9']} | {row['0.8']} | "
            f"{replication_differences[corpus]:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "A causal predictive span is a useful intervention coordinate, not necessarily a unique feature basis. Full-state replacement tests recoverability of an entire carrier; router-span replacement tests the supervised directions used by one linear chart. Threshold crossings summarize a smooth curve and should not be promoted to a universal anatomical boundary."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "exact_errors": exact_errors,
        "replication": replication_differences,
        "crossings": crossings,
        "layer16": {corpus: row["16"] for corpus, row in results.items()},
        "layer4": {corpus: row["4"] for corpus, row in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
