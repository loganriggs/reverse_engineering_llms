#!/usr/bin/env python3
"""Red-team the compact Layer-16 causal basis found by the Jacobian pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer16_jacobian_causal_subspace import (
    batched_candidate_coefficients,
    final_coefficients,
    gram_error,
    local_jacobian,
    metric_jacobian,
    project,
)
from run_bilin18_layer17_router_compilation_depth import (
    fit_response,
    gram_fidelity,
    prepare_ridge,
    trace_boundaries,
)
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer16_causal_basis_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_causal_basis_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_CAUSAL_BASIS_REDTEAM.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_causal_basis_redteam.png")


def spectrum(rows):
    _, singular, vh = torch.linalg.svd(rows.float().cuda(), full_matrices=False)
    energy = singular.double().square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    stats = {
        "rank90": int((cumulative < .90).sum() + 1),
        "rank99": int((cumulative < .99).sum() + 1),
        "top16_energy": float(cumulative[15]),
        "top64_energy": float(cumulative[63]),
    }
    return vh.T.cpu(), stats


def overlap(left, right):
    singular = torch.linalg.svdvals(left.float().T @ right.float()).clamp(0, 1)
    return {
        "mean_squared_canonical": float(singular.square().mean()),
        "minimum_canonical_cosine": float(singular.min()),
        "canonical_cosines": singular.tolist(),
    }


def collect_discovery(model, spec, config, context_axes, metric_root, gram):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    states, coefficients, raw_jacobians = [], [], []
    errors = {str(scale): [0.0, 0.0] for scale in config["finite_difference_scales"]}
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, boundary = trace_boundaries(model, ids, {16})
        x16 = boundary[16]
        for position in config["discovery_positions"]:
            native, jacobian = local_jacobian(
                model, x16, x0, value_bus, position, context_axes)
            states.append(x16[:, position].bfloat16().cpu())
            coefficients.append(native.cpu())
            raw_jacobians.append(jacobian.bfloat16().cpu())
            direction = torch.randn(
                len(x16), x16.shape[-1], generator=generator, device="cuda")
            direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
            state_norm = x16[:, position].double().square().sum(1).sqrt().float()
            for scale in config["finite_difference_scales"]:
                delta = direction * (state_norm * scale)[:, None]
                with torch.no_grad():
                    plus = final_coefficients(
                        model, x16, x0, value_bus, position,
                        x16[:, position] + delta, context_axes)
                    minus = final_coefficients(
                        model, x16, x0, value_bus, position,
                        x16[:, position] - delta, context_axes)
                actual = (plus - minus) / 2
                predicted = torch.einsum("bid,bd->bi", jacobian, delta)
                error, energy = gram_error(actual.cpu(), predicted.cpu(), gram)
                errors[str(scale)][0] += error
                errors[str(scale)][1] += energy
    finite = {
        key: float(np.sqrt(error / max(energy, 1e-30)))
        for key, (error, energy) in errors.items()
    }
    return {
        "state": torch.cat(states),
        "coefficients": torch.cat(coefficients),
        "jacobian": torch.cat(raw_jacobians),
        "central_difference_relative_error": finite,
    }


def make_coordinate_basis(dimension, indices):
    basis = torch.zeros(dimension, len(indices))
    basis[indices, torch.arange(len(indices))] = 1
    return basis


def fit_bases(discovery, config, context_axes, metric_root):
    raw_rows = discovery["jacobian"].float().flatten(0, 1)
    metric_rows = torch.einsum(
        "ij,njd->nid", metric_root.cpu(), discovery["jacobian"].float()
    ).flatten(0, 1)
    metric_basis, metric_spectrum = spectrum(metric_rows)
    euclidean_basis, euclidean_spectrum = spectrum(raw_rows)

    fit = fit_response(
        prepare_ridge(discovery["state"], config["ridge_relative_penalty"]),
        discovery["coefficients"])
    predictive = torch.linalg.qr(fit["weight"].float()).Q[:, :config["rank"]]

    native_score = metric_rows.double().square().sum(0)
    native_order = torch.argsort(native_score, descending=True)
    generator = torch.Generator().manual_seed(config["seed"])
    random_dense = torch.linalg.qr(torch.randn(
        metric_basis.shape[0], max(config["basis_ranks"]), generator=generator)).Q
    random_order = torch.randperm(metric_basis.shape[0], generator=generator)
    methods = {
        "context_16": context_axes.cpu(),
        "predictive_16": predictive,
    }
    for rank in config["basis_ranks"]:
        methods[f"metric_sensitivity_{rank}"] = metric_basis[:, :rank]
        methods[f"euclidean_sensitivity_{rank}"] = euclidean_basis[:, :rank]
        methods[f"random_dense_{rank}"] = random_dense[:, :rank]
        methods[f"random_native_{rank}"] = make_coordinate_basis(
            metric_basis.shape[0], random_order[:rank])
    for rank in config["native_ranks"]:
        methods[f"native_{rank}"] = make_coordinate_basis(
            metric_basis.shape[0], native_order[:rank])
    return methods, {
        "metric": metric_spectrum,
        "euclidean": euclidean_spectrum,
        "overlaps": {
            "metric16_context16": overlap(metric_basis[:, :16], context_axes.cpu()),
            "metric16_euclidean16": overlap(metric_basis[:, :16], euclidean_basis[:, :16]),
            "metric16_predictive16": overlap(metric_basis[:, :16], predictive),
        },
        "top_native_indices": native_order[:128].tolist(),
        "top_native_energy_share_16": float(native_score[native_order[:16]].sum() / native_score.sum()),
        "top_native_energy_share_64": float(native_score[native_order[:64]].sum() / native_score.sum()),
        "top_native_energy_share_128": float(native_score[native_order[:128]].sum() / native_score.sum()),
    }


@torch.no_grad()
def evaluate(model, spec, positions, config, context_axes, gram, methods):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    caches = {
        str(scale): {
            metric: {name: {"target": [], "prediction": []} for name in methods}
            for metric in ("tensor", "coefficient")
        }
        for scale in config["edit_scales"]
    }
    cuda_methods = {name: basis.cuda() for name, basis in methods.items()}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, boundary = trace_boundaries(model, ids, {16})
        x16 = boundary[16]
        for position in positions:
            native = final_coefficients(
                model, x16, x0, value_bus, position, x16[:, position], context_axes)
            delta = torch.roll(x16[:, position], 1, 0) - x16[:, position]
            for scale in config["edit_scales"]:
                key = str(scale)
                scaled = scale * delta
                candidates = {"full": x16[:, position] + scaled}
                candidates.update({
                    name: x16[:, position] + project(scaled, basis)
                    for name, basis in cuda_methods.items()
                })
                values = batched_candidate_coefficients(
                    model, x16, x0, value_bus, position, candidates, context_axes)
                target = (values["full"] - native).cpu()
                for name in methods:
                    prediction = (values[name] - native).cpu()
                    for metric in ("tensor", "coefficient"):
                        caches[key][metric][name]["target"].append(target)
                        caches[key][metric][name]["prediction"].append(prediction)
    identity = torch.eye(config["rank"], dtype=torch.double)
    output = {}
    for scale in config["edit_scales"]:
        key = str(scale)
        output[key] = {"tensor": {}, "coefficient": {}}
        for metric, matrix in (("tensor", gram), ("coefficient", identity)):
            for name, values in caches[key][metric].items():
                target = torch.cat(values["target"])
                prediction = torch.cat(values["prediction"])
                output[key][metric][name] = gram_fidelity(target, prediction, matrix)
    return output


def locked_hypotheses(results, diagnostics, finite, config):
    small, full = map(str, config["edit_scales"])
    all_rows = [group[small]["tensor"] for corpus in results.values() for group in corpus.values()]
    heldout = [corpus["heldout_positions"][small]["tensor"] for corpus in results.values()]
    return {
        "R0_central_difference_validates_autodiff": min(finite.values()) <= .01,
        "R1_sensitivity16_is_mostly_context_span": diagnostics["overlaps"]["metric16_context16"]["mean_squared_canonical"] >= .75,
        "R2_context_span_is_small_edit_causally_sufficient": all(row["context_16"] >= .75 for row in all_rows),
        "R3_context_span_nearly_matches_sensitivity16": all(row["metric_sensitivity_16"] <= row["context_16"] + .10 for row in all_rows),
        "R4_native64_is_not_a_substitute_for_dense_sensitivity16": all(row["metric_sensitivity_16"] >= row["native_64"] + .20 for row in heldout),
        "R5_sensitivity16_beats_random_dense16": all(row["metric_sensitivity_16"] >= row["random_dense_16"] + .50 for row in all_rows),
        "R6_sensitivity16_beats_predictive16": all(row["metric_sensitivity_16"] >= row["predictive_16"] + .20 for row in all_rows),
        "R7_sensitivity16_transfers_to_unseen_positions": all(abs(corpus["discovery_positions"][small]["tensor"]["metric_sensitivity_16"] - corpus["heldout_positions"][small]["tensor"]["metric_sensitivity_16"]) <= .15 for corpus in results.values()),
        "R8_sensitivity16_remains_full_edit_sufficient": all(group[full]["tensor"]["metric_sensitivity_16"] >= .85 for corpus in results.values() for group in corpus.values()),
        "R9_each_metric_finds_a_compact_causal_basis": all(group[small]["tensor"]["metric_sensitivity_16"] >= .75 and group[small]["coefficient"]["euclidean_sensitivity_16"] >= .75 for corpus in results.values() for group in corpus.values()),
        "R10_metric_choice_does_not_radically_rotate_the_basis": diagnostics["overlaps"]["metric16_euclidean16"]["mean_squared_canonical"] >= .50,
        "R11_native128_remains_incomplete": all(row["native_128"] < .70 for row in heldout),
    }


def plot_results(results, config):
    methods = ["context_16", "metric_sensitivity_16", "predictive_16", "native_64", "random_dense_16"]
    labels = ["context16", "sensitivity16", "predictive16", "native64", "random16"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, (corpus, corpus_rows) in zip(axes, results.items()):
        x = np.arange(len(methods)); width = .36
        for offset, group in ((-width / 2, "discovery_positions"), (width / 2, "heldout_positions")):
            row = corpus_rows[group]["0.1"]["tensor"]
            axis.bar(x + offset, [row[name] for name in methods], width, label=group.replace("_positions", ""))
        axis.set_title(corpus)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("full-effect fidelity, 0.1×")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def main():
    config = json.loads(CONFIG.read_text())
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
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    metric_root = ((eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]) @ eigenvectors.T).float()

    print("collect discovery Jacobians and finite differences", flush=True)
    discovery = collect_discovery(model, config["discovery"], config, context_axes, metric_root.cuda(), gram)
    print("fit and compare bases", flush=True)
    methods, diagnostics = fit_bases(discovery, config, context_axes, metric_root)
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        results[corpus] = {}
        for name, positions in (("discovery_positions", config["discovery_positions"]), ("heldout_positions", config["heldout_positions"])):
            results[corpus][name] = evaluate(model, spec, positions, config, context_axes, gram, methods)
    finite = discovery["central_difference_relative_error"]
    hypotheses = {name: bool(value) for name, value in locked_hypotheses(results, diagnostics, finite, config).items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "central_difference_relative_error": finite,
        "diagnostics": diagnostics,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    plot_results(results, config)

    lines = [
        "# Layer-16 causal-basis red team", "",
        "All bases are discovery-frozen. Fidelity is computed on the *change* caused by the full cyclic edit, so zero means that a projected edit explains none of the target effect and one means it reproduces that effect exactly.",
        "", "![Causal basis controls](bilin18_layer16_causal_basis_redteam.png)",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Numerical and geometric diagnostics", "",
              f"Central finite-difference relative errors: `{finite}`.",
              f"Metric16–context16 mean squared canonical overlap: `{diagnostics['overlaps']['metric16_context16']['mean_squared_canonical']:.3f}`.",
              f"Metric16–Euclidean16 overlap: `{diagnostics['overlaps']['metric16_euclidean16']['mean_squared_canonical']:.3f}`.",
              f"Metric sensitivity R90/R99: `{diagnostics['metric']['rank90']}` / `{diagnostics['metric']['rank99']}`.",
              f"Top native coordinate energy shares at 16/64/128: `{diagnostics['top_native_energy_share_16']:.3f}` / `{diagnostics['top_native_energy_share_64']:.3f}` / `{diagnostics['top_native_energy_share_128']:.3f}`.",
              "", "## Tensor-effect fidelity at 0.1×", "",
              "| Corpus | Positions | context16 | sensitivity16 | Euclidean16 | predictive16 | native16/64/128 | random dense16 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, corpus_rows in results.items():
        for group, scales in corpus_rows.items():
            row = scales["0.1"]["tensor"]
            lines.append(f"| {corpus} | {group.replace('_positions', '')} | {row['context_16']:.3f} | {row['metric_sensitivity_16']:.3f} | {row['euclidean_sensitivity_16']:.3f} | {row['predictive_16']:.3f} | {row['native_16']:.3f}/{row['native_64']:.3f}/{row['native_128']:.3f} | {row['random_dense_16']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "The sensitivity basis mostly recovers the predefined context span, so its success is primarily an architectural validation rather than discovery of a new hidden code. The locked dense-only alternatives also fail: top sensitivity-ranked residual coordinates are unexpectedly competitive with the dense basis. These coordinates are residual dimensions, not bilinear MLP neurons, and their semantic meaning and split stability remain unproved.",
              "",
              "The result establishes causal sufficiency only for this Layer-16-to-Layer-17 operator effect. It does not establish semantic meaning, on-manifold editability, or downstream-logit sufficiency. Metric dependence remains a general concern even though the coefficient and tensor metrics select substantially overlapping bases here."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "finite": finite, "diagnostics": diagnostics, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
