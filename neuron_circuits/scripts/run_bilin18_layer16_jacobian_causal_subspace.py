#!/usr/bin/env python3
"""Find causal Layer-16 state subspaces from the exact final-suffix Jacobian."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_router_compilation_depth import (
    fit_response,
    gram_fidelity,
    prepare_ridge,
    trace_boundaries,
)
from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer16_jacobian_causal_subspace.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_jacobian_causal_subspace.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER16_JACOBIAN_CAUSAL_SUBSPACE.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer16_jacobian_causal_subspace.png")


def final_coefficients(model, x16, x0, value_bus, position, z, context_axes):
    dimension = model.config.n_embd
    edited = x16.clone()
    edited[:, position] = z
    block = model.blocks[17]
    mixed = block.lambdas[0].float() * edited + block.lambdas[1].float() * x0
    attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    state = F.rms_norm(mixed + attention.float(), (dimension,))
    return state[:, position] @ context_axes


def local_jacobian(model, x16, x0, value_bus, position, context_axes):
    with torch.enable_grad():
        z = x16[:, position].detach().clone().requires_grad_(True)
        output = final_coefficients(
            model, x16.detach(), x0.detach(), value_bus.detach(), position, z, context_axes)
        rows = []
        for coordinate in range(output.shape[1]):
            gradient, = torch.autograd.grad(
                output[:, coordinate].sum(), z,
                retain_graph=coordinate + 1 < output.shape[1])
            rows.append(gradient.detach())
    return output.detach(), torch.stack(rows, dim=1)


def metric_jacobian(jacobian, metric_root):
    return torch.einsum("ij,bjd->bid", metric_root, jacobian)


def project(delta, basis):
    return (delta @ basis) @ basis.T


def site_project(delta, jacobian_metric):
    basis = torch.linalg.qr(jacobian_metric.transpose(1, 2)).Q
    return torch.einsum("bd,bdr,ber->be", delta, basis, basis)


@torch.no_grad()
def batched_candidate_coefficients(
        model, x16, x0, value_bus, position, candidates, context_axes):
    names = list(candidates)
    batch = len(x16)
    z = torch.cat([candidates[name] for name in names])
    repeats = len(names)
    values = final_coefficients(
        model, x16.repeat(repeats, 1, 1), x0.repeat(repeats, 1, 1),
        value_bus.repeat(*([repeats] + [1] * (value_bus.ndim - 1))),
        position, z, context_axes)
    return {name: values[index * batch:(index + 1) * batch]
            for index, name in enumerate(names)}


def gram_error(target, prediction, gram):
    target = target.double(); prediction = prediction.double(); difference = target - prediction
    return (
        float(torch.einsum("bi,ij,bj->", difference, gram, difference)),
        float(torch.einsum("bi,ij,bj->", target, gram, target)),
    )


def collect_discovery(model, spec, config, context_axes, metric_root, gram):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    states, coefficients, jacobians = [], [], []
    finite_error = 0.0; finite_energy = 0.0
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, boundary = trace_boundaries(model, ids, {16})
        x16 = boundary[16]
        for position in config["positions"]:
            native, jacobian = local_jacobian(
                model, x16, x0, value_bus, position, context_axes)
            states.append(x16[:, position].bfloat16().cpu())
            coefficients.append(native.cpu())
            # Preserve the dynamic range of rare high-sensitivity sites.  An
            # earlier compilation-depth run showed that IEEE float16 can
            # overflow on otherwise valid late-layer states; bfloat16 keeps
            # the compact cache without imposing that narrow exponent range.
            jacobians.append(metric_jacobian(jacobian, metric_root).bfloat16().cpu())

            direction = torch.randn(
                len(x16), x16.shape[-1], generator=generator, device="cuda")
            scale = (x16[:, position].double().square().mean(1).sqrt()
                     * config["finite_difference_scale"]).float()
            delta = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
            delta = delta * scale[:, None] * np.sqrt(x16.shape[-1])
            with torch.no_grad():
                actual = final_coefficients(
                    model, x16, x0, value_bus, position,
                    x16[:, position] + delta, context_axes) - native
            predicted = torch.einsum("bid,bd->bi", jacobian, delta)
            error, energy = gram_error(actual.cpu(), predicted.cpu(), gram)
            finite_error += error; finite_energy += energy
    return {
        "state": torch.cat(states),
        "coefficients": torch.cat(coefficients),
        "jacobian_metric": torch.cat(jacobians),
        "finite_difference_relative_error": float(np.sqrt(finite_error / max(finite_energy, 1e-30))),
    }


def fit_bases(discovery, config):
    print("fit predictive span", flush=True)
    fit = fit_response(
        prepare_ridge(discovery["state"], config["ridge_relative_penalty"]),
        discovery["coefficients"])
    predictive = torch.linalg.qr(fit["weight"].float()).Q[:, :config["rank"]]
    print("svd sensitivity rows", flush=True)
    rows = discovery["jacobian_metric"].float().cuda().flatten(0, 1)
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    sensitivity = vh.T.cpu()
    generator = torch.Generator().manual_seed(config["seed"])
    random = torch.linalg.qr(torch.randn(
        sensitivity.shape[0], max(config["sensitivity_ranks"]), generator=generator)).Q
    energy = singular.double().square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    return predictive, sensitivity, random, {
        "rank90": int((cumulative < .90).sum() + 1),
        "rank99": int((cumulative < .99).sum() + 1),
        "top16_energy": float(cumulative[15]),
        "top64_energy": float(cumulative[63]),
    }


def evaluate(model, spec, config, context_axes, metric_root, gram,
             predictive, sensitivity, random):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    methods = ["linearization", "predictive_16", "site_specific"]
    methods += [f"sensitivity_{rank}" for rank in config["sensitivity_ranks"]]
    methods += ["random_16", "random_64"]
    effects = {
        str(scale): {name: {"target": [], "prediction": []} for name in methods}
        for scale in config["edit_scales"]
    }
    effect_norms = {str(scale): [] for scale in config["edit_scales"]}
    predictive = predictive.cuda(); sensitivity = sensitivity.cuda(); random = random.cuda()
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, boundary = trace_boundaries(model, ids, {16})
        x16 = boundary[16]
        for position in config["positions"]:
            native, jacobian = local_jacobian(
                model, x16, x0, value_bus, position, context_axes)
            jacobian_metric = metric_jacobian(jacobian, metric_root)
            delta = torch.roll(x16[:, position], 1, 0) - x16[:, position]
            for scale in config["edit_scales"]:
                key = str(scale)
                scaled = scale * delta
                candidates = {
                    "full": x16[:, position] + scaled,
                    "predictive_16": x16[:, position] + project(scaled, predictive),
                    "site_specific": x16[:, position] + site_project(scaled, jacobian_metric),
                    "random_16": x16[:, position] + project(scaled, random[:, :16]),
                    "random_64": x16[:, position] + project(scaled, random[:, :64]),
                }
                for rank in config["sensitivity_ranks"]:
                    candidates[f"sensitivity_{rank}"] = (
                        x16[:, position] + project(scaled, sensitivity[:, :rank]))
                values = batched_candidate_coefficients(
                    model, x16, x0, value_bus, position, candidates, context_axes)
                target = values["full"] - native
                linear = torch.einsum("bid,bd->bi", jacobian, scaled)
                effect_norms[key].append(target.cpu())
                effects[key]["linearization"]["target"].append(target.cpu())
                effects[key]["linearization"]["prediction"].append(linear.cpu())
                for name in methods:
                    if name == "linearization":
                        continue
                    effects[key][name]["target"].append(target.cpu())
                    effects[key][name]["prediction"].append((values[name] - native).cpu())
    output = {}
    for scale in config["edit_scales"]:
        key = str(scale)
        output[key] = {}
        for name, rows_by_kind in effects[key].items():
            target = torch.cat(rows_by_kind["target"])
            prediction = torch.cat(rows_by_kind["prediction"])
            output[key][name] = gram_fidelity(target, prediction, gram)
        targets = torch.cat(effect_norms[key]).double()
        output[key]["effect_operator_rms"] = float(
            (torch.einsum("bi,ij,bj->", targets, gram, targets) / len(targets)).sqrt())
        output[key]["examples"] = len(targets)
    return output


def plot_results(results, config):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ranks = config["sensitivity_ranks"]
    for axis, scale in zip(axes, config["edit_scales"]):
        key = str(scale)
        for corpus, row in results.items():
            axis.plot(ranks, [row[key][f"sensitivity_{rank}"] for rank in ranks],
                      marker="o", label=corpus)
        axis.set_title(f"{scale:g}× cyclic edit")
        axis.set_xlabel("shared sensitivity rank")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("full-effect fidelity")
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
    metric_root = (
        (eigenvectors * eigenvalues.clamp_min(0).sqrt()[None]) @ eigenvectors.T
    ).float()

    print("collect discovery jacobians", flush=True)
    discovery = collect_discovery(
        model, config["discovery"], config, context_axes, metric_root.cuda(), gram)
    predictive, sensitivity, random, spectrum = fit_bases(discovery, config)
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, context_axes, metric_root.cuda(), gram,
            predictive, sensitivity, random)

    small, full = map(str, config["edit_scales"])
    hypotheses = {
        "J0_autodiff_jacobian_matches_finite_difference": (
            discovery["finite_difference_relative_error"] <= .01),
        "J1_small_edit_linearization_is_accurate": all(
            row[small]["linearization"] >= .90 for row in results.values()),
        "J2_full_edit_is_more_nonlinear": all(
            row[full]["linearization"] <= row[small]["linearization"] - .10
            for row in results.values()),
        "J3_site_specific_rowspace_captures_small_effect": all(
            row[small]["site_specific"] >= .90 for row in results.values()),
        "J4_site_specific_rowspace_remains_useful_for_full_edits": all(
            row[full]["site_specific"] >= .70 for row in results.values()),
        "J5_shared_sensitivity64_captures_small_effect": all(
            row[small]["sensitivity_64"] >= .70 for row in results.values()),
        "J6_sensitivity64_beats_random64": all(
            row[small]["sensitivity_64"] >= row[small]["random_64"] + .30
            for row in results.values()),
        "J7_sensitivity16_beats_predictive16": all(
            row[small]["sensitivity_16"] >= row[small]["predictive_16"] + .20
            for row in results.values()),
        "J8_sensitivity128_is_useful_for_full_edits": all(
            row[full]["sensitivity_128"] >= .50 for row in results.values()),
        "J9_shared_sensitivity_subspace_transfers_ood": (
            max(row[small]["sensitivity_64"] for row in results.values())
            - min(row[small]["sensitivity_64"] for row in results.values()) <= .15),
        "J10_tensor_metric_is_positive": float(eigenvalues.min()) >= -1e-8,
    }
    hypotheses = {name: bool(value) for name, value in hypotheses.items()}
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "operator_gram_min_eigenvalue": float(eigenvalues.min()),
        "finite_difference_relative_error": discovery["finite_difference_relative_error"],
        "sensitivity_spectrum": spectrum,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    plot_results(results, config)

    lines = [
        "# Exact Layer-16 Jacobian causal subspaces", "",
        "The output metric is induced by the contracted context-to-operator tensor. Shared sensitivity bases come from the SVD of discovery `G^(1/2)J` rows; every finite patch is executed through native final attention and RMS normalization.",
        "", "![Jacobian causal subspace](bilin18_layer16_jacobian_causal_subspace.png)",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Finite-difference relative error: `{discovery['finite_difference_relative_error']:.3e}`. Sensitivity R90/R99: `{spectrum['rank90']}` / `{spectrum['rank99']}`.",
              "", "## Full-effect fidelity", "",
              "| Corpus | Scale | linear J | predictive16 | site-specific | sensitivity 16/32/64/128 | random 16/64 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, corpus_rows in results.items():
        for scale in config["edit_scales"]:
            row = corpus_rows[str(scale)]
            ranks = "/".join(f"{row[f'sensitivity_{rank}']:.3f}" for rank in config["sensitivity_ranks"])
            lines.append(
                f"| {corpus} | {scale:g} | {row['linearization']:.3f} | "
                f"{row['predictive_16']:.3f} | {row['site_specific']:.3f} | {ranks} | "
                f"{row['random_16']:.3f}/{row['random_64']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "The exact Jacobian describes one local tangent map. Its site-specific row space is an oracle causal coordinate for infinitesimal edits, while a discovery-aggregated sensitivity space tests reusable geometry. Degradation at full edit scale measures curvature and context change; it does not invalidate the derivative. The failed predictive span remains a valid decoder even when it is a poor intervention basis."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "finite_difference_error": discovery["finite_difference_relative_error"],
        "spectrum": spectrum,
        "results": results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
