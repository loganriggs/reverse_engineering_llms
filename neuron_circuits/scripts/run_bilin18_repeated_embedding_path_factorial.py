#!/usr/bin/env python3
"""Factor current-token effects across the repeated embedding pathways."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_tangent_bundle_quotient import build_role_tensor
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_repeated_embedding_path_factorial.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_path_factorial.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_REPEATED_EMBEDDING_PATH_FACTORIAL.md")
FIGURE = Path("artifacts/bilin18_tensor_decomposition/bilin18_repeated_embedding_path_factorial.png")
SOURCES = ("initial", "bus", "injection")


def source_bus(model, x0):
    block = model.blocks[0]
    mixed = block.lambdas[0].float() * x0 + block.lambdas[1].float() * x0
    normalized = F.rms_norm(mixed, (mixed.shape[-1],))
    return block.attn.c_v(normalized).view(
        len(x0), x0.shape[1], block.attn.n_head, block.attn.head_dim)


@torch.no_grad()
def execute_variants(model, native_x0, donor_x0, native_bus, donor_bus,
                     variants, position, context_axes):
    """Run variants with independent initial, persistent-bus, and injection sources."""
    names = list(variants)
    initial = torch.cat([
        donor_x0 if variants[name]["initial"] else native_x0 for name in names
    ])
    bus = torch.cat([
        donor_bus if variants[name]["bus"] else native_bus for name in names
    ])
    x = initial
    operator = None
    dimension = x.shape[-1]
    for layer, block in enumerate(model.blocks):
        injected = torch.cat([
            donor_x0 if layer in variants[name]["injection_layers"] else native_x0
            for name in names
        ])
        mixed = block.lambdas[0].float() * x + block.lambdas[1].float() * injected
        attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
        pre = mixed + attention.float()
        state = F.rms_norm(pre, (dimension,))
        if layer == 17:
            operator = state[:, position] @ context_axes
        x = pre + block.mlp(state).float()
    final = F.rms_norm(x, (dimension,))[:, position]
    raw = model.lm_head(final).float()
    logits = 30 * torch.tanh(raw / 30)
    batch = len(native_x0)
    return {
        name: {
            "operator": operator[index * batch:(index + 1) * batch],
            "logits": logits[index * batch:(index + 1) * batch],
        }
        for index, name in enumerate(names)
    }


@torch.no_grad()
def standard_execute(model, x0, position, context_axes):
    x = x0; bus = None; operator = None; dimension = x.shape[-1]
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0].float() * x + block.lambdas[1].float() * x0
        attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
        pre = mixed + attention.float(); state = F.rms_norm(pre, (dimension,))
        if layer == 17:
            operator = state[:, position] @ context_axes
        x = pre + block.mlp(state).float()
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return operator, 30 * torch.tanh(raw / 30)


def factorial_variants():
    return {
        "".join(str(int(value)) for value in bits): {
            "initial": bits[0], "bus": bits[1],
            "injection_layers": set(range(18)) if bits[2] else set(),
        }
        for bits in itertools.product((False, True), repeat=3)
    }


def depth_variants(groups):
    variants = {
        "none": {"initial": False, "bus": False, "injection_layers": set()},
        "all": {"initial": False, "bus": False, "injection_layers": set(range(18))},
    }
    for start, stop in groups:
        variants[f"group_{start}_{stop}"] = {
            "initial": False, "bus": False,
            "injection_layers": set(range(start, stop + 1)),
        }
        variants[f"prefix_0_{stop}"] = {
            "initial": False, "bus": False,
            "injection_layers": set(range(0, stop + 1)),
        }
        variants[f"suffix_{start}_17"] = {
            "initial": False, "bus": False,
            "injection_layers": set(range(start, 18)),
        }
    return variants


def empty_metric(names):
    return {
        name: {metric: {"error": 0.0, "target": 0.0}
               for metric in ("operator", "fisher", "cross_entropy", "donor_kl")}
        for name in names
    }


def update_metrics(accumulator, native, target, variants, labels, gram):
    target_operator = target["operator"] - native["operator"]
    target_logits = target["logits"] - native["logits"]
    probability = native["logits"].softmax(-1)
    target_centered = target_logits - (probability * target_logits).sum(-1, keepdim=True)
    native_ce = F.cross_entropy(native["logits"], labels, reduction="none")
    target_ce = F.cross_entropy(target["logits"], labels, reduction="none") - native_ce
    target_probability = target["logits"].log_softmax(-1).exp()
    native_log_probability = native["logits"].log_softmax(-1)
    target_log_probability = target["logits"].log_softmax(-1)
    target_kl = (target_probability * (target_log_probability - native_log_probability)).sum(-1)
    operator_energy = torch.einsum("bi,ij,bj->", target_operator.double(), gram, target_operator.double())
    fisher_energy = (probability * target_centered.square()).double().sum()
    ce_energy = target_ce.double().square().sum()
    kl_energy = target_kl.double().sum()
    for name, value in variants.items():
        prediction_operator = value["operator"] - native["operator"]
        prediction_logits = value["logits"] - native["logits"]
        error_operator = prediction_operator - target_operator
        error_logits = prediction_logits - target_logits
        error_centered = error_logits - (probability * error_logits).sum(-1, keepdim=True)
        prediction_ce = F.cross_entropy(value["logits"], labels, reduction="none") - native_ce
        variant_log_probability = value["logits"].log_softmax(-1)
        residual_kl = (target_probability * (target_log_probability - variant_log_probability)).sum(-1)
        rows = accumulator[name]
        rows["operator"]["error"] += float(torch.einsum("bi,ij,bj->", error_operator.double(), gram, error_operator.double()))
        rows["operator"]["target"] += float(operator_energy)
        rows["fisher"]["error"] += float((probability * error_centered.square()).double().sum())
        rows["fisher"]["target"] += float(fisher_energy)
        rows["cross_entropy"]["error"] += float((prediction_ce - target_ce).double().square().sum())
        rows["cross_entropy"]["target"] += float(ce_energy)
        rows["donor_kl"]["error"] += float(residual_kl.double().sum())
        rows["donor_kl"]["target"] += float(kl_energy)


def finish_metrics(accumulator):
    return {
        name: {
            metric: 1 - values["error"] / max(values["target"], 1e-30)
            for metric, values in rows.items()
        }
        for name, rows in accumulator.items()
    }


def shapley_vectors(outputs, field):
    result = {}
    for source_index, source in enumerate(SOURCES):
        value = 0
        other = [index for index in range(3) if index != source_index]
        for size in range(3):
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            for subset in itertools.combinations(other, size):
                absent = [0, 0, 0]
                for index in subset:
                    absent[index] = 1
                present = absent.copy(); present[source_index] = 1
                key0 = "".join(map(str, absent)); key1 = "".join(map(str, present))
                value = value + weight * (outputs[key1][field] - outputs[key0][field])
        result[source] = value
    return result


def empty_shapley():
    return {
        metric: {source: {"inner": 0.0, "energy": 0.0} for source in SOURCES}
        for metric in ("operator", "fisher")
    } | {"target": {"operator": 0.0, "fisher": 0.0}, "recomposition_error": 0.0}


def update_shapley(accumulator, outputs, gram):
    target_operator = outputs["111"]["operator"] - outputs["000"]["operator"]
    target_logits = outputs["111"]["logits"] - outputs["000"]["logits"]
    probability = outputs["000"]["logits"].softmax(-1)
    target_centered = target_logits - (probability * target_logits).sum(-1, keepdim=True)
    operator_phi = shapley_vectors(outputs, "operator")
    logit_phi = shapley_vectors(outputs, "logits")
    recomposed = sum(operator_phi.values())
    accumulator["recomposition_error"] += float((recomposed - target_operator).double().square().sum())
    operator_target_energy = torch.einsum("bi,ij,bj->", target_operator.double(), gram, target_operator.double())
    fisher_target_energy = (probability * target_centered.square()).double().sum()
    accumulator["target"]["operator"] += float(operator_target_energy)
    accumulator["target"]["fisher"] += float(fisher_target_energy)
    for source in SOURCES:
        phi = operator_phi[source]
        accumulator["operator"][source]["inner"] += float(torch.einsum("bi,ij,bj->", phi.double(), gram, target_operator.double()))
        accumulator["operator"][source]["energy"] += float(torch.einsum("bi,ij,bj->", phi.double(), gram, phi.double()))
        phi_logits = logit_phi[source]
        phi_centered = phi_logits - (probability * phi_logits).sum(-1, keepdim=True)
        accumulator["fisher"][source]["inner"] += float((probability * phi_centered * target_centered).double().sum())
        accumulator["fisher"][source]["energy"] += float((probability * phi_centered.square()).double().sum())


def finish_shapley(accumulator):
    output = {"recomposition_squared_error": accumulator["recomposition_error"]}
    for metric in ("operator", "fisher"):
        target = max(accumulator["target"][metric], 1e-30)
        output[metric] = {
            source: {
                "projection_share": row["inner"] / target,
                "energy_ratio": row["energy"] / target,
                "cosine": row["inner"] / max(math.sqrt(row["energy"] * target), 1e-30),
            }
            for source, row in accumulator[metric].items()
        }
    return output


@torch.no_grad()
def evaluate_corpus(model, spec, config, context_axes, gram):
    source = np.load(spec["path"], mmap_mode="r")
    rows = np.asarray(source[spec["start"]:spec["start"] + spec["rows"]])
    factorial = factorial_variants(); depth = depth_variants(config["layer_groups"])
    factorial_metrics = empty_metric(factorial)
    depth_metrics = empty_metric(depth)
    shapley = empty_shapley()
    endpoint = {
        field: {"error": 0.0, "energy": 0.0}
        for field in ("operator", "logits")
    }
    dimension = model.config.n_embd
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        native_x0 = F.rms_norm(model.embedding(ids), (dimension,)).float()
        native_bus = source_bus(model, native_x0)
        for position in config["positions"]:
            donor_x0 = native_x0.clone()
            donor_x0[:, position] = torch.roll(native_x0[:, position], 1, 0)
            donor_bus = source_bus(model, donor_x0)
            outputs = execute_variants(
                model, native_x0, donor_x0, native_bus, donor_bus,
                factorial, position, context_axes)
            native_reference = standard_execute(model, native_x0, position, context_axes)
            donor_reference = standard_execute(model, donor_x0, position, context_axes)
            for key, reference in (("000", native_reference), ("111", donor_reference)):
                for field, expected in (("operator", reference[0]), ("logits", reference[1])):
                    difference = outputs[key][field] - expected
                    endpoint[field]["error"] += float(difference.double().square().sum())
                    endpoint[field]["energy"] += float(expected.double().square().sum())
            labels = torch.as_tensor(raw[:, position + 1], device="cuda")
            update_metrics(factorial_metrics, outputs["000"], outputs["111"], outputs, labels, gram)
            update_shapley(shapley, outputs, gram)

            depth_outputs = execute_variants(
                model, native_x0, donor_x0, native_bus, donor_bus,
                depth, position, context_axes)
            update_metrics(depth_metrics, depth_outputs["none"], depth_outputs["all"], depth_outputs, labels, gram)
    return {
        "endpoint_relative_error": {
            field: math.sqrt(values["error"] / max(values["energy"], 1e-30))
            for field, values in endpoint.items()
        },
        "factorial_fidelity": finish_metrics(factorial_metrics),
        "injection_depth_fidelity": finish_metrics(depth_metrics),
        "shapley": finish_shapley(shapley),
    }


def source_order(row, metric):
    return tuple(sorted(SOURCES, key=lambda source: row["shapley"][metric][source]["projection_share"], reverse=True))


def locked_hypotheses(results, config):
    factorial = [row["factorial_fidelity"] for row in results.values()]
    depth = [row["injection_depth_fidelity"] for row in results.values()]
    return {
        "X0_factorial_endpoints_are_exact": max(
            value for row in results.values()
            for value in row["endpoint_relative_error"].values()) <= 1e-6,
        "X1_repeated_injection_is_operator_relevant": all(row["001"]["operator"] >= .50 for row in factorial),
        "X2_repeated_injection_is_behavior_relevant": all(row["001"]["fisher"] >= .30 for row in factorial),
        "X3_initial_state_alone_is_not_sufficient": all(row["100"]["operator"] < .50 and row["100"]["fisher"] < .50 for row in factorial),
        "X4_persistent_value_bus_is_not_sufficient": all(row["010"]["operator"] < .50 and row["010"]["fisher"] < .50 for row in factorial),
        "X5_no_single_three_layer_injection_group_is_sufficient": all(max(row[f"group_{start}_{stop}"]["operator"] for start, stop in config["layer_groups"]) < .80 for row in depth),
        "X6_late_injections_are_not_the_whole_history": all(row["group_15_17"]["operator"] <= .90 for row in depth),
        "X7_injection_shapley_is_positive": all(row["shapley"][metric]["injection"]["projection_share"] > 0 for row in results.values() for metric in ("operator", "fisher")),
        "X8_pathway_attribution_is_not_metric_invariant": any(abs(row["shapley"]["operator"][source]["projection_share"] - row["shapley"]["fisher"][source]["projection_share"]) >= .10 for row in results.values() for source in SOURCES),
        "X9_source_ordering_transfers_ood": all(len({source_order(row, metric) for row in results.values()}) == 1 for metric in ("operator", "fisher")),
        "X10_cross_entropy_does_not_reverse_injection_relevance": all(row["001"]["cross_entropy"] > 0 for row in factorial),
    }


def plot_results(results):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(results)); labels = list(results)
    width = .22
    for index, source in enumerate(SOURCES):
        axes[0].bar(x + (index - 1) * width, [row["shapley"]["operator"][source]["projection_share"] for row in results.values()], width, label=source)
        axes[1].bar(x + (index - 1) * width, [row["shapley"]["fisher"][source]["projection_share"] for row in results.values()], width, label=source)
    axes[0].set_title("Layer-17 operator")
    axes[1].set_title("final-logit Fisher geometry")
    for axis in axes:
        axis.set_xticks(x, labels); axis.axhline(0, color="black", linewidth=.7); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Shapley projection share")
    axes[1].legend(frameon=False)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)


def main():
    config = json.loads(CONFIG.read_text())
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False)
    context_axes = core["context"][:, :config["rank"]].float().cuda()
    tangent_axes = core["tangent"][:, :config["rank"]].float().cuda()
    output_axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor, *_ = build_role_tensor(model.blocks[17], context_axes, tangent_axes, output_axes)
    flat = tensor.reshape(-1, config["rank"]).double()
    gram = (flat.T @ flat).cuda()
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        results[corpus] = evaluate_corpus(model, spec, config, context_axes, gram)
    hypotheses = {name: bool(value) for name, value in locked_hypotheses(results, config).items()}
    payload = {"config": config, "block_lambdas": [[float(value) for value in block.lambdas.float().cpu()] for block in model.blocks], "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)); plot_results(results)

    lines = ["# Repeated-embedding path factorial", "",
             "A donor current-token embedding is introduced independently through the initial residual, the persistent Layer-0 value bus, and the `x0` injection repeated at every block. `001` means injection only; `010` bus only; `100` initial state only; `111` is the complete donor-token substitution.", "",
             "![Repeated-embedding Shapley attribution](bilin18_repeated_embedding_path_factorial.png)", "", "## Locked hypotheses", ""]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## Source coalitions", "", "| Corpus | endpoint error | coalition | operator | Fisher logits | CE effect | donor-KL recovery |", "|---|---:|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for key in ("100", "010", "001", "110", "101", "011"):
            values = row["factorial_fidelity"][key]
            endpoint = max(row["endpoint_relative_error"].values())
            lines.append(f"| {corpus} | {endpoint:.2e} | `{key}` | {values['operator']:.3f} | {values['fisher']:.3f} | {values['cross_entropy']:.3f} | {values['donor_kl']:.3f} |")
    lines += ["", "## Shapley projection shares", "", "| Corpus | metric | initial | persistent bus | repeated injection |", "|---|---|---:|---:|---:|"]
    for corpus, row in results.items():
        for metric in ("operator", "fisher"):
            shares = row["shapley"][metric]
            lines.append(f"| {corpus} | {metric} | {shares['initial']['projection_share']:.3f} | {shares['bus']['projection_share']:.3f} | {shares['injection']['projection_share']:.3f} |")
    lines += ["", "## Injection depth", "", "| Corpus | group | operator | Fisher logits |", "|---|---|---:|---:|"]
    for corpus, row in results.items():
        depth = row["injection_depth_fidelity"]
        for start, stop in config["layer_groups"]:
            name = f"group_{start}_{stop}"; lines.append(f"| {corpus} | {start}-{stop} only | {depth[name]['operator']:.3f} | {depth[name]['fisher']:.3f} |")
        for name in ("prefix_0_2", "prefix_0_5", "prefix_0_8", "prefix_0_11", "prefix_0_14", "all"):
            lines.append(f"| {corpus} | {name} | {depth[name]['operator']:.3f} | {depth[name]['fisher']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "This factorial distinguishes architectural entry paths for one finite token-embedding substitution. It does not make the three paths statistically independent: their nonlinear interactions are allocated by Shapley averaging. Operator and Fisher shares answer different questions and need not agree. Hybrid coalitions are causal interventions, not necessarily states encountered on natural text."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
