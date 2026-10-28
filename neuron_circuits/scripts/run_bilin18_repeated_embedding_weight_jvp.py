#!/usr/bin/env python3
"""Exact weight-derived JVP decomposition of repeated embedding paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_repeated_embedding_weight_jvp.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_weight_jvp.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_REPEATED_EMBEDDING_WEIGHT_JVP.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_weight_jvp.png")


def source_names(config):
    return ["initial", "bus"] + [f"inject_{start}_{stop}" for start, stop in config["layer_groups"]]


def replace_position(base, position, value):
    output = base.clone(); output[:, position] = value
    return output


def source_bus(model, x0):
    block = model.blocks[0]
    mixed = (block.lambdas[0] + block.lambdas[1]).float() * x0
    normalized = F.rms_norm(mixed, (mixed.shape[-1],))
    return block.attn.c_v(normalized).view(
        len(x0), x0.shape[1], block.attn.n_head, block.attn.head_dim)


def make_function(model, native_x0, position, context_axes, groups):
    dimension = native_x0.shape[-1]

    def function(*aliases):
        initial_x0 = replace_position(native_x0, position, aliases[0])
        bus_x0 = replace_position(native_x0, position, aliases[1])
        injections = [replace_position(native_x0, position, value) for value in aliases[2:]]
        x = initial_x0; bus = source_bus(model, bus_x0); operator = None
        for layer, block in enumerate(model.blocks):
            group_index = next(index for index, (start, stop) in enumerate(groups) if start <= layer <= stop)
            x0 = injections[group_index]
            mixed = block.lambdas[0].float() * x + block.lambdas[1].float() * x0
            attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
            pre = mixed + attention.float(); state = F.rms_norm(pre, (dimension,))
            if layer == 17:
                operator = state[:, position] @ context_axes
            x = pre + block.mlp(state).float()
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
        return operator, 30 * torch.tanh(raw / 30)

    return function


def metric_inner(metric, left, right, native_logits, gram):
    if metric == "operator":
        return torch.einsum("bi,ij,bj->", left.double(), gram, right.double())
    probability = native_logits.softmax(-1)
    left = left - (probability * left).sum(-1, keepdim=True)
    right = right - (probability * right).sum(-1, keepdim=True)
    return (probability * left * right).double().sum()


def ce_derivative(tangent, logits, labels):
    probability = logits.softmax(-1)
    return (probability * tangent).sum(-1) - tangent.gather(1, labels[:, None]).squeeze(1)


def methods_from_paths(paths, config):
    names = source_names(config); output = {name: paths[name] for name in names}
    injection_names = names[2:]
    output["injection_all"] = sum(paths[name] for name in injection_names)
    for index, (_, stop) in enumerate(config["layer_groups"]):
        output[f"injection_prefix_0_{stop}"] = sum(paths[name] for name in injection_names[:index + 1])
    output["initial_plus_bus"] = paths["initial"] + paths["bus"]
    return output


def empty_accumulator(methods, scales):
    return {
        "path": {
            metric: {name: {"inner": 0.0, "energy": 0.0, "error": 0.0}
                     for name in methods}
            for metric in ("operator", "fisher", "cross_entropy")
        },
        "target": {metric: 0.0 for metric in ("operator", "fisher", "cross_entropy")},
        "finite": {
            str(scale): {metric: {"error": 0.0, "target": 0.0}
                         for metric in ("operator", "fisher", "cross_entropy")}
            for scale in scales
        },
        "recomposition": {"operator_error": 0.0, "operator_target": 0.0,
                          "logit_error": 0.0, "logit_target": 0.0},
    }


def update_accumulator(acc, primal, total, paths, finite, labels, gram, config):
    operator, logits = primal; total_operator, total_logits = total
    operator_energy = metric_inner("operator", total_operator, total_operator, logits, gram)
    fisher_energy = metric_inner("fisher", total_logits, total_logits, logits, gram)
    total_ce = ce_derivative(total_logits, logits, labels)
    ce_energy = total_ce.double().square().sum()
    for metric, energy in (("operator", operator_energy), ("fisher", fisher_energy), ("cross_entropy", ce_energy)):
        acc["target"][metric] += float(energy)

    operator_paths = methods_from_paths({name: value[0] for name, value in paths.items()}, config)
    logit_paths = methods_from_paths({name: value[1] for name, value in paths.items()}, config)
    for name in operator_paths:
        for metric, prediction, target in (
            ("operator", operator_paths[name], total_operator),
            ("fisher", logit_paths[name], total_logits),
        ):
            inner = metric_inner(metric, prediction, target, logits, gram)
            energy = metric_inner(metric, prediction, prediction, logits, gram)
            error = metric_inner(metric, prediction - target, prediction - target, logits, gram)
            row = acc["path"][metric][name]
            row["inner"] += float(inner); row["energy"] += float(energy); row["error"] += float(error)
        prediction_ce = ce_derivative(logit_paths[name], logits, labels)
        row = acc["path"]["cross_entropy"][name]
        row["inner"] += float((prediction_ce * total_ce).double().sum())
        row["energy"] += float(prediction_ce.double().square().sum())
        row["error"] += float((prediction_ce - total_ce).double().square().sum())

    recomposed_operator = sum(value[0] for value in paths.values())
    recomposed_logits = sum(value[1] for value in paths.values())
    acc["recomposition"]["operator_error"] += float((recomposed_operator - total_operator).double().square().sum())
    acc["recomposition"]["operator_target"] += float(total_operator.double().square().sum())
    acc["recomposition"]["logit_error"] += float((recomposed_logits - total_logits).double().square().sum())
    acc["recomposition"]["logit_target"] += float(total_logits.double().square().sum())

    for scale, output in finite.items():
        finite_operator = output[0] - operator; finite_logits = output[1] - logits
        prediction_operator = scale * total_operator; prediction_logits = scale * total_logits
        finite_ce = F.cross_entropy(output[1], labels, reduction="none") - F.cross_entropy(logits, labels, reduction="none")
        prediction_ce = scale * total_ce
        for metric, prediction, target in (
            ("operator", prediction_operator, finite_operator),
            ("fisher", prediction_logits, finite_logits),
        ):
            row = acc["finite"][str(scale)][metric]
            row["error"] += float(metric_inner(metric, prediction - target, prediction - target, logits, gram))
            row["target"] += float(metric_inner(metric, target, target, logits, gram))
        row = acc["finite"][str(scale)]["cross_entropy"]
        row["error"] += float((prediction_ce - finite_ce).double().square().sum())
        row["target"] += float(finite_ce.double().square().sum())


def finish(acc):
    output = {"paths": {}, "finite_fidelity": {}, "recomposition_relative_error": {}}
    for metric, rows in acc["path"].items():
        target = max(acc["target"][metric], 1e-30)
        output["paths"][metric] = {
            name: {
                "projection_share": row["inner"] / target,
                "energy_ratio": row["energy"] / target,
                "cosine": row["inner"] / max(math.sqrt(row["energy"] * target), 1e-30),
                "fidelity": 1 - row["error"] / target,
            }
            for name, row in rows.items()
        }
    for scale, rows in acc["finite"].items():
        output["finite_fidelity"][scale] = {
            metric: 1 - row["error"] / max(row["target"], 1e-30)
            for metric, row in rows.items()
        }
    for field in ("operator", "logit"):
        output["recomposition_relative_error"][field] = math.sqrt(
            acc["recomposition"][f"{field}_error"] /
            max(acc["recomposition"][f"{field}_target"], 1e-30))
    return output


def warm_rotary(model, x0):
    with torch.no_grad():
        x = x0; bus = None; dimension = x.shape[-1]
        for block in model.blocks:
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
            pre = mixed + attention; x = pre + block.mlp(F.rms_norm(pre, (dimension,)))


def evaluate(model, spec, config, context_axes, gram, donor_shift=1):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    names = source_names(config)
    method_names = list(methods_from_paths({name: 0 for name in names}, config))
    acc = empty_accumulator(method_names, config["finite_scales"])
    dimension = model.config.n_embd
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)).float().detach()
        warm_rotary(model, x0)
        for position in config["positions"]:
            base = x0[:, position].clone()
            donor = torch.roll(base, donor_shift, 0)
            delta = donor - base
            primals = tuple(base.clone() for _ in names)
            function = make_function(model, x0, position, context_axes, config["layer_groups"])
            all_tangents = tuple(delta.clone() for _ in names)
            primal, total = torch.func.jvp(function, primals, all_tangents)
            paths = {}
            zero = torch.zeros_like(delta)
            for index, name in enumerate(names):
                tangents = tuple(delta if offset == index else zero for offset in range(len(names)))
                _, paths[name] = torch.func.jvp(function, primals, tangents)
            finite = {}
            for scale in config["finite_scales"]:
                arguments = tuple(base + scale * delta for _ in names)
                finite[scale] = function(*arguments)
            labels = torch.as_tensor(raw[:, position + 1], device="cuda")
            update_accumulator(acc, primal, total, paths, finite, labels, gram, config)
    return finish(acc)


def locked_hypotheses(results, config):
    metrics = ("operator", "fisher")
    injection_names = [f"inject_{start}_{stop}" for start, stop in config["layer_groups"]]
    return {
        "W0_weight_path_jvps_recompose": max(value for row in results.values() for value in row["recomposition_relative_error"].values()) <= 1e-5,
        "W1_one_percent_edit_is_locally_linear": all(row["finite_fidelity"]["0.01"][metric] >= .99 for row in results.values() for metric in metrics),
        "W2_full_token_edit_is_materially_nonlinear": all(any(row["finite_fidelity"]["1.0"][metric] <= row["finite_fidelity"]["0.01"][metric] - .10 for metric in metrics) for row in results.values()),
        "W3_repeated_injection_dominates_local_sensitivity": all(row["paths"][metric]["injection_all"]["projection_share"] >= .80 for row in results.values() for metric in metrics),
        "W4_initial_and_bus_are_minor_local_paths": all(row["paths"][metric]["initial_plus_bus"]["projection_share"] <= .20 for row in results.values() for metric in metrics),
        "W5_layers_0_through_11_capture_most_local_injection_effect": all(row["paths"][metric]["injection_prefix_0_11"]["fidelity"] >= .80 for row in results.values() for metric in metrics),
        "W6_layers_15_through_17_are_locally_minor": all(abs(row["paths"][metric]["inject_15_17"]["projection_share"]) <= .10 for row in results.values() for metric in metrics),
        "W7_injection_dominance_is_metric_robust": all(abs(row["paths"]["operator"]["injection_all"]["projection_share"] - row["paths"]["fisher"]["injection_all"]["projection_share"]) <= .10 for row in results.values()),
        "W8_injection_dominance_transfers_ood": all(max(row["paths"][metric]["injection_all"]["projection_share"] for row in results.values()) - min(row["paths"][metric]["injection_all"]["projection_share"] for row in results.values()) <= .15 for metric in metrics),
        "W9_true_token_loss_uses_the_injection_path": all(row["paths"]["cross_entropy"]["injection_all"]["projection_share"] > 0 for row in results.values()),
        "W10_equal_direct_coefficients_do_not_imply_equal_path_importance": all(any(max(row["paths"][metric][name]["projection_share"] for name in injection_names) - min(row["paths"][metric][name]["projection_share"] for name in injection_names) >= .10 for metric in metrics) for row in results.values()),
    }


def plot_results(results, config):
    names = [f"inject_{start}_{stop}" for start, stop in config["layer_groups"]]
    labels = [name.replace("inject_", "") for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(names)); width = .24
    for offset, (corpus, row) in enumerate(results.items()):
        for axis, metric in zip(axes, ("operator", "fisher")):
            axis.bar(x + (offset - 1) * width, [row["paths"][metric][name]["projection_share"] for name in names], width, label=corpus)
    for axis, metric in zip(axes, ("operator", "fisher")):
        axis.set_title(metric); axis.set_xticks(x, labels, rotation=35, ha="right"); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("local projection share"); axes[1].legend(frameon=False)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)


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
    flat = tensor.reshape(-1, config["rank"]).double(); gram = (flat.T @ flat).cuda()
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        results[corpus] = evaluate(model, spec, config, context_axes, gram)
    hypotheses = {name: bool(value) for name, value in locked_hypotheses(results, config).items()}
    payload = {"config": config, "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)); plot_results(results, config)

    lines = ["# Repeated-embedding weight-path JVP", "",
             "Each number is obtained by differentiating the actual weight graph at native text states. The eight independently perturbable aliases are the initial state, persistent value-bus source, and six three-layer `x0` injection groups.", "", "![Weight-path shares](bilin18_repeated_embedding_weight_jvp.png)", "", "## Locked hypotheses", ""]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Path projection shares", "", "| Corpus | metric | initial | bus | injection all | inject 0-2/3-5/6-8/9-11/12-14/15-17 |", "|---|---|---:|---:|---:|---:|"]
    injection_names = [f"inject_{start}_{stop}" for start, stop in config["layer_groups"]]
    for corpus, row in results.items():
        for metric in ("operator", "fisher", "cross_entropy"):
            paths = row["paths"][metric]
            groups = "/".join(f"{paths[name]['projection_share']:.3f}" for name in injection_names)
            lines.append(f"| {corpus} | {metric} | {paths['initial']['projection_share']:.3f} | {paths['bus']['projection_share']:.3f} | {paths['injection_all']['projection_share']:.3f} | {groups} |")
    lines += ["", "## Finite radius of the local explanation", "", "| Corpus | scale | operator | Fisher | CE effect |", "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for scale, values in row["finite_fidelity"].items():
            lines.append(f"| {corpus} | {scale} | {values['operator']:.3f} | {values['fisher']:.3f} | {values['cross_entropy']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "This is the direct weight-and-connection answer for one donor-token tangent at each native state. It removes probe fitting and Shapley-allocation ambiguity locally. It remains conditional on the state, donor direction, and output metric; finite-edit degradation quantifies where that local ontology stops being sufficient."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
