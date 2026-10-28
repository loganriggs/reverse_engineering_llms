#!/usr/bin/env python3
"""Red-team dense margin channels by separating top-1 and runner-up logits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer17_backward_mlp_native_channels import (
    factor_source,
    fidelity,
    final_mlp_channel_adjoints,
    layer16_direct_write,
    summarize,
    support_summary,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_margin_logit_channel_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_margin_logit_channel_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_MARGIN_LOGIT_CHANNEL_REDTEAM.md")
COMPONENTS = ("top1_logit", "runnerup_logit", "margin")


def channel_contributions(model, row17, position, factor_data, indices, weights):
    token_indices = {
        "top1_logit": factor_data["top1"][indices],
        "runnerup_logit": factor_data["top2"][indices],
    }
    adjoints = final_mlp_channel_adjoints(
        model.blocks[17], row17, position, {},
        lm_head=model.lm_head, token_indices=token_indices)
    adjoints["margin"] = adjoints["top1_logit"] - adjoints["runnerup_logit"]
    direct_write = layer16_direct_write(factor_data, indices, weights)
    contributions = {
        component: torch.einsum("bcd,bnd->bcn", adjoint, direct_write).cpu()
        for component, adjoint in adjoints.items()
    }
    return contributions, direct_write


def discovery_order(model, spec, factor_data, path_data, weights, config):
    source = load_rows(spec)
    channel_count = model.blocks[17].mlp.Left.weight.shape[0]
    dot = {name: torch.zeros(channel_count, dtype=torch.float64) for name in COMPONENTS}
    square = {name: torch.zeros(channel_count, dtype=torch.float64) for name in COMPONENTS}
    maximum_error = 0.0
    offset = spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        for position in config["positions"]:
            indices = slice(offset + consumed, offset + consumed + len(ids))
            contributions, direct_write = channel_contributions(
                model, row17, position, factor_data, indices, weights)
            saved_margin = path_data["margin"][indices, 2].float().cuda()
            saved_target = torch.einsum("bd,bnd->bn", saved_margin, direct_write).cpu()
            margin_error = float(
                (contributions["margin"].sum(1).double() - saved_target.double()).norm()
                / saved_target.double().norm().clamp_min(1e-30))
            maximum_error = max(maximum_error, margin_error)
            for component in COMPONENTS:
                target = contributions[component].sum(1)
                dot[component] += torch.einsum(
                    "bcn,bn->c", contributions[component].double(), target.double())
                square[component] += contributions[component].double().square().sum((0, 2))
            consumed += len(ids)
    ordering = {
        component: torch.argsort(2 * dot[component] - square[component], descending=True)
        for component in COMPONENTS
    }
    return ordering, maximum_error


def evaluate(model, spec, factor_data, path_data, weights, ordering, config):
    source = load_rows(spec)
    individual_budgets = config["budgets_per_logit"]
    margin_budgets = config["direct_margin_budgets"]
    store = {
        component: {"target": [], "predictions": {}, "mass": []}
        for component in COMPONENTS
    }
    for component in ("top1_logit", "runnerup_logit"):
        store[component]["predictions"] = {budget: [] for budget in individual_budgets}
    store["margin"]["predictions"] = {budget: [] for budget in margin_budgets}
    union_predictions = {budget: [] for budget in individual_budgets}
    maximum_error = 0.0
    offset = spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        for position in config["positions"]:
            indices = slice(offset + consumed, offset + consumed + len(ids))
            contributions, direct_write = channel_contributions(
                model, row17, position, factor_data, indices, weights)
            saved_margin = path_data["margin"][indices, 2].float().cuda()
            saved_target = torch.einsum("bd,bnd->bn", saved_margin, direct_write).cpu()
            maximum_error = max(maximum_error, float(
                (contributions["margin"].sum(1).double() - saved_target.double()).norm()
                / saved_target.double().norm().clamp_min(1e-30)))
            for component in COMPONENTS:
                target = contributions[component].sum(1)
                store[component]["target"].append(target)
                _, _, mass = support_summary(contributions[component], target)
                store[component]["mass"].extend(mass.tolist())
            for component in ("top1_logit", "runnerup_logit"):
                for budget in individual_budgets:
                    selected = ordering[component][:budget]
                    store[component]["predictions"][budget].append(
                        contributions[component][:, selected].sum(1))
            for budget in margin_budgets:
                selected = ordering["margin"][:budget]
                store["margin"]["predictions"][budget].append(
                    contributions["margin"][:, selected].sum(1))
            for budget in individual_budgets:
                selected = torch.unique(torch.cat((
                    ordering["top1_logit"][:budget],
                    ordering["runnerup_logit"][:budget],
                )))
                union_predictions[budget].append(
                    contributions["margin"][:, selected].sum(1))
            consumed += len(ids)

    results = {}
    for component in ("top1_logit", "runnerup_logit"):
        target = torch.cat(store[component]["target"])
        curve = {
            str(budget): fidelity(target, torch.cat(store[component]["predictions"][budget]))
            for budget in individual_budgets
        }
        k90 = next((budget for budget in individual_budgets if curve[str(budget)] >= .90), None)
        results[component] = {
            "fidelity": curve,
            "k90": k90,
            "absolute_projection_mass": summarize(store[component]["mass"]),
        }
    margin_target = torch.cat(store["margin"]["target"])
    margin_curve = {
        str(budget): fidelity(margin_target, torch.cat(store["margin"]["predictions"][budget]))
        for budget in margin_budgets
    }
    union_curve = {
        str(budget): fidelity(margin_target, torch.cat(union_predictions[budget]))
        for budget in individual_budgets
    }
    results["margin"] = {
        "fidelity": margin_curve,
        "k90": next((budget for budget in margin_budgets if margin_curve[str(budget)] >= .90), None),
        "logit_union_fidelity": union_curve,
        "absolute_projection_mass": summarize(store["margin"]["mass"]),
    }
    return results, maximum_error


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    paths = torch.load(config["path_path"], map_location="cpu", weights_only=False)
    weights = factors["layer16_weights"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    discovery_spec = config["selection_discovery"]
    discovery_data = factor_source(factors, discovery_spec)
    ordering, maximum_error = discovery_order(
        model, discovery_spec, discovery_data,
        paths[discovery_spec["factor_dataset"]], weights, config)
    overlap128 = len(set(ordering["top1_logit"][:128].tolist())
                     & set(ordering["runnerup_logit"][:128].tolist()))
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = factor_source(factors, spec)
        rows, error = evaluate(
            model, spec, data, paths[spec["factor_dataset"]],
            weights, ordering, config)
        maximum_error = max(maximum_error, error)
        results[corpus] = {"sites": spec["rows"] * len(config["positions"]),
                           "components": rows}
    del model
    torch.cuda.empty_cache()

    def numeric_k90(value):
        return value if value is not None else 4609

    hypotheses = {
        "R0_margin_equals_top1_minus_runnerup": maximum_error <= 1e-5,
        "R1_top1_logit_path_is_native_sparse": all(
            row["components"]["top1_logit"]["fidelity"]["128"] >= .90
            for row in results.values()),
        "R2_runnerup_logit_path_is_native_sparse": all(
            row["components"]["runnerup_logit"]["fidelity"]["128"] >= .90
            for row in results.values()),
        "R3_margin_density_is_created_by_combining_logit_roles": all(
            numeric_k90(row["components"]["margin"]["k90"])
            >= 4 * max(
                numeric_k90(row["components"]["top1_logit"]["k90"]),
                numeric_k90(row["components"]["runnerup_logit"]["k90"]),
            ) for row in results.values()),
        "R4_union_of_logit_banks_compresses_margin": all(
            row["components"]["margin"]["logit_union_fidelity"]["128"] >= .90
            for row in results.values()),
        "R5_logit_banks_are_distinct": overlap128 < 96,
        "R6_margin_cancellation_exceeds_individual_logits": all(
            row["components"]["margin"]["absolute_projection_mass"]["median"]
            >= max(
                row["components"]["top1_logit"]["absolute_projection_mass"]["median"],
                row["components"]["runnerup_logit"]["absolute_projection_mass"]["median"],
            ) + 1 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["R0_margin_equals_top1_minus_runnerup"]["maximum_relative_error"] = maximum_error
    hypothesis_rows["R5_logit_banks_are_distinct"]["top128_overlap"] = overlap128
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "top128_overlap": overlap128,
        "orderings": {name: value.tolist() for name, value in ordering.items()},
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Top-1 versus runner-up channel decomposition of the dense margin path",
        "",
        "The model-selected margin is decomposed exactly into its top-1 logit path "
        "minus its runner-up logit path. Each individual channel bank and their "
        "frozen union are selected on FineWeb discovery and evaluated OOD.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Maximum margin recomposition error: `{maximum_error:.2e}`. Top-128 bank "
        f"overlap: `{overlap128}` channels.",
        "",
        "| Corpus | Top1 K32/K64/K128/K256/K512 (K90) | Runner K32/K64/K128/K256/K512 (K90) | Margin direct K128/K512/K2048/K4096 (K90) | Margin union 16/32/64/128/256/512 per logit | Median mass top1/runner/margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        components = row["components"]
        top = components["top1_logit"]
        runner = components["runnerup_logit"]
        margin = components["margin"]
        individual = (32, 64, 128, 256, 512)
        direct = (128, 512, 2048, 4096)
        union = config["budgets_per_logit"]
        lines.append(
            f"| {corpus} | "
            + "/".join(f"{top['fidelity'][str(k)]:.3f}" for k in individual)
            + f" ({top['k90']}) | "
            + "/".join(f"{runner['fidelity'][str(k)]:.3f}" for k in individual)
            + f" ({runner['k90']}) | "
            + "/".join(f"{margin['fidelity'][str(k)]:.3f}" for k in direct)
            + f" ({margin['k90']}) | "
            + "/".join(f"{margin['logit_union_fidelity'][str(k)]:.3f}" for k in union)
            + " | "
            + f"{top['absolute_projection_mass']['median']:.2f}/"
            f"{runner['absolute_projection_mass']['median']:.2f}/"
            f"{margin['absolute_projection_mass']['median']:.2f} |")
    lines += [
        "",
        "If individual logits are sparse but their union is not sufficient, margin "
        "density comes from context-varying signed interaction rather than merely "
        "combining two fixed sparse output banks.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
