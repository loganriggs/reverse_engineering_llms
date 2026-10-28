#!/usr/bin/env python3
"""Optimize endpoint-relative Tucker cores inside the fixed tangent quotient."""

from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch

from run_bilin18_layer17_backward_mlp_native_channels import factor_source, fidelity
from run_bilin18_layer17_backward_mlp_symmetric_tensor import collect_sites
from run_bilin18_layer17_tangent_bundle_quotient import enrich_sites, slice_sites
from run_bilin18_layer17_tangent_bundle_core_stability import unfold_modes
from run_bilin18_layer17_tangent_bundle_split_redteam import projector_overlap
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_tangent_bundle_behavior_core.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_behavior_core.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_BEHAVIOR_CORE.md")
CHECKPOINT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_behavior_core.pt")


def orthogonal(raw):
    return torch.linalg.qr(raw, mode="reduced").Q


def core_tensor(tensor, output, tangent, context):
    first = torch.einsum("oa,odx->adx", output, tensor)
    second = torch.einsum("db,adx->abx", tangent, first)
    return torch.einsum("xc,abx->abc", context, second)


def coefficients(sites, base, components, device):
    return {
        "context": (sites["state"].float() @ base["context_basis"].float()).to(device),
        "tangent": torch.einsum(
            "bnd,dk->bnk", sites["tangent"].float(), base["tangent_basis"].float()).to(device),
        "adjoint": {
            component: (sites["adjoint"][component].float() @ base["output_basis"].float()).to(device)
            for component in components
        },
        "target": {
            component: torch.einsum(
                "bd,bnd->bn", sites["adjoint"][component].float(), sites["output"].float()).to(device)
            for component in components
        },
    }


def predict(data, tensor, bases, component):
    output, tangent, context = bases
    core = core_tensor(tensor, output, tangent, context)
    a = data["adjoint"][component] @ output
    d = data["tangent"] @ tangent
    x = data["context"] @ context
    return torch.einsum("bo,bnd,odx,bx->bn", a, d, core, x)


def normalized_loss(data, tensor, bases, components):
    losses = []
    for component in components:
        target = data["target"][component]
        prediction = predict(data, tensor, bases, component)
        losses.append(
            (target - prediction).double().square().sum()
            / target.double().square().sum().clamp_min(1e-30))
    return torch.stack(losses).mean().float()


@torch.inference_mode()
def evaluate(data, tensor, bases, components):
    return {
        component: fidelity(
            data["target"][component].cpu(),
            predict(data, tensor, bases, component).cpu())
        for component in components
    }


def initialize(modes, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for role in ("output", "tangent", "context"):
        initial = modes[role][:, :rank].clone()
        if seed:
            initial += .03 * torch.randn(initial.shape, generator=generator)
        rows.append(initial.cuda().requires_grad_(True))
    return rows


def optimize(fit, validation, tensor, modes, rank, components, seed, config):
    raw = initialize(modes, rank, seed)
    optimizer = torch.optim.Adam(raw, lr=config["learning_rate"])
    best = None
    trace = []
    for step in range(config["steps"] + 1):
        bases = tuple(orthogonal(value) for value in raw)
        if step % config["validation_interval"] == 0:
            with torch.no_grad():
                fit_loss = float(normalized_loss(fit, tensor, bases, components))
                validation_loss = float(normalized_loss(validation, tensor, bases, components))
            trace.append({"step": step, "fit_loss": fit_loss, "validation_loss": validation_loss})
            if best is None or validation_loss < best["validation_loss"]:
                best = {
                    "step": step,
                    "validation_loss": validation_loss,
                    "fit_loss": fit_loss,
                    "bases": tuple(value.detach().cpu().clone() for value in bases),
                }
        if step == config["steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_loss(fit, tensor, bases, components)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw, 1.0)
        optimizer.step()
    return best, trace


def objective_mean(row, components):
    return float(np.mean([row[component] for component in components]))


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    base = torch.load(config["base_tensor_path"], map_location="cpu", weights_only=False)
    tensor = base["tensor"].float().cuda()
    modes, _ = unfold_modes(base["tensor"])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    block = model.blocks[17]
    spec = config["discovery"]
    data = factor_source(factors, spec)
    discovery = enrich_sites(
        block, collect_sites(model, spec, data, config), data, spec,
        factors["layer16_weights"], config["evaluation_batch_size"])
    fit_sites = config["fit_sites"]
    fit = coefficients(
        slice_sites(discovery, slice(0, fit_sites)), base, config["components"], "cuda")
    validation = coefficients(
        slice_sites(discovery, slice(fit_sites, len(discovery["state"]))),
        base, config["components"], "cuda")
    external = {}
    for corpus, corpus_spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        corpus_data = factor_source(factors, corpus_spec)
        sites = enrich_sites(
            block, collect_sites(model, corpus_spec, corpus_data, config),
            corpus_data, corpus_spec, factors["layer16_weights"],
            config["evaluation_batch_size"])
        external[corpus] = coefficients(sites, base, config["components"], "cuda")

    rank = config["rank"]
    hosvd_bases = tuple(modes[role][:, :rank].float().cuda() for role in ("output", "tangent", "context"))
    baseline = {
        corpus: evaluate(rows, tensor, hosvd_bases, config["components"])
        for corpus, rows in {"fit": fit, "validation": validation, **external}.items()
    }
    runs, saved = {}, {}
    for objective, components in config["objectives"].items():
        runs[objective] = {}
        for seed in config["optimization_seeds"]:
            print("optimize", objective, seed, flush=True)
            best, trace = optimize(
                fit, validation, tensor, modes, rank, components, seed, config)
            bases = tuple(value.float().cuda() for value in best["bases"])
            evaluations = {
                corpus: evaluate(rows, tensor, bases, config["components"])
                for corpus, rows in {"fit": fit, "validation": validation, **external}.items()
            }
            runs[objective][str(seed)] = {
                "best_step": best["step"],
                "fit_loss": best["fit_loss"],
                "validation_loss": best["validation_loss"],
                "trace": trace,
                "evaluation": evaluations,
            }
            saved[f"{objective}_{seed}"] = tuple(value.cpu() for value in bases)

    eval_names = tuple(config["evaluation"])
    def external_objective(row, components):
        return float(np.mean([
            row["evaluation"][corpus][component]
            for corpus in eval_names for component in components]))
    baseline_scores = float(np.mean([
        baseline[corpus][component] for corpus in eval_names
        for component in config["objectives"]["scores"]]))
    baseline_margin = float(np.mean([baseline[corpus]["margin"] for corpus in eval_names]))
    score_values = {
        seed: external_objective(row, config["objectives"]["scores"])
        for seed, row in runs["scores"].items()
    }
    all_margin_values = {
        seed: float(np.mean([
            row["evaluation"][corpus]["margin"] for corpus in eval_names]))
        for seed, row in runs["all"].items()
    }
    best_validation = {
        objective: min(rows, key=lambda seed: rows[seed]["validation_loss"])
        for objective, rows in runs.items()
    }
    selected_score = saved[f"scores_{best_validation['scores']}"]
    selected_all = saved[f"all_{best_validation['all']}"]
    cross_objective_overlap = {
        role: projector_overlap(selected_score[index], selected_all[index])
        for index, role in enumerate(("output", "tangent", "context"))
    }
    within_objective_overlap = {}
    for objective in config["objectives"]:
        pair_rows = []
        for first, second in combinations((str(seed) for seed in config["optimization_seeds"]), 2):
            first_bases = saved[f"{objective}_{first}"]
            second_bases = saved[f"{objective}_{second}"]
            pair_rows.append({
                "seeds": [first, second],
                **{
                    role: projector_overlap(first_bases[index], second_bases[index])
                    for index, role in enumerate(("output", "tangent", "context"))
                },
            })
        within_objective_overlap[objective] = pair_rows
    hypotheses = {
        "O0_hosvd_baseline_matches_prior_result": (
            abs(baseline["fineweb"]["prediction"] - .891) <= .01
            and abs(baseline["fineweb"]["target"] - .860) <= .01
            and abs(baseline["wikitext"]["prediction"] - .921) <= .01
            and abs(baseline["tinystories"]["prediction"] - .942) <= .01),
        "O1_score_objective_improves_scores": all(
            value >= baseline_scores + .02 for value in score_values.values()),
        "O2_all_objective_improves_margin": all(
            value >= baseline_margin + .08 for value in all_margin_values.values()),
        "O3_margin_gain_does_not_destroy_scores": all(
            row["evaluation"][corpus][component] >= .80
            for row in runs["all"].values() for corpus in eval_names
            for component in ("prediction", "target")),
        "O4_validation_selection_transfers": all(
            external_objective(
                runs[objective][best_validation[objective]], components)
            >= max(external_objective(row, components) for row in runs[objective].values()) - .03
            for objective, components in config["objectives"].items()),
        "O5_behavior_cores_are_seed_stable": (
            max(score_values.values()) - min(score_values.values()) <= .05
            and max(all_margin_values.values()) - min(all_margin_values.values()) <= .05),
        "O6_score_and_margin_cores_are_distinct": min(cross_objective_overlap.values()) < .90,
        "O7_optimized_core_is_a_heldout_claim": (
            fit_sites == 192 and len(validation["context"]) == 64
            and spec["start"] + spec["rows"] <= config["evaluation"]["fineweb"]["start"]),
    }
    torch.save({"base": base, "optimized_bases": saved, "best_validation": best_validation}, CHECKPOINT)
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "baseline": baseline,
        "baseline_external_scores": baseline_scores,
        "baseline_external_margin": baseline_margin,
        "score_external_objectives": score_values,
        "all_external_margin": all_margin_values,
        "best_validation_seed": best_validation,
        "cross_objective_overlap": cross_objective_overlap,
        "within_objective_overlap": within_objective_overlap,
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Behavior-optimized rank-16 cores of the tangent-bundle tensor", "",
        "Two endpoint definitions are optimized inside the same fixed rank-64 role tensor: individual prediction/target scores, and those scores plus their precision-sensitive difference. Training uses 192 discovery sites; checkpoint selection uses the remaining 64; external corpora are untouched evaluation.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += [
        "", f"HOSVD mean external score fidelity: `{baseline_scores:.3f}`; margin: `{baseline_margin:.3f}`.",
        "", "## External objective summary", "",
        "| Objective | seed | selected step | validation loss | mean external score | mean external margin |", "|---|---:|---:|---:|---:|---:|",
    ]
    for objective, seeds in runs.items():
        for seed, row in seeds.items():
            score = external_objective(row, config["objectives"]["scores"])
            margin = float(np.mean([row["evaluation"][corpus]["margin"] for corpus in eval_names]))
            lines.append(f"| {objective} | {seed} | {row['best_step']} | {row['validation_loss']:.4f} | {score:.3f} | {margin:.3f} |")
    lines += ["", "## Selected-core external fidelity", "",
              "Each cell is prediction/target/margin.", "",
              "| Objective | corpus | HOSVD-16 | optimized-16 |", "|---|---|---:|---:|"]
    for objective in config["objectives"]:
        row = runs[objective][best_validation[objective]]
        for corpus in eval_names:
            base_cell = "/".join(f"{baseline[corpus][c]:.3f}" for c in config["components"])
            opt_cell = "/".join(f"{row['evaluation'][corpus][c]:.3f}" for c in config["components"])
            lines.append(f"| {objective} | {corpus} | {base_cell} | {opt_cell} |")
    lines += [
        "", "Selected score/all core projector overlap (output/tangent/context): "
        + "/".join(f"{cross_objective_overlap[role]:.3f}" for role in ("output", "tangent", "context")) + ".",
        "", "Exploratory within-objective pairwise overlap ranges (output/tangent/context):",
    ]
    for objective, pair_rows in within_objective_overlap.items():
        ranges = []
        for role in ("output", "tangent", "context"):
            values = [row[role] for row in pair_rows]
            ranges.append(f"{min(values):.3f}–{max(values):.3f}")
        lines.append(f"- {objective}: " + "/".join(ranges))
    lines += [
        "", "If endpoint-specific optimization yields different transferable cores, compactness is real but the shortest useful program depends on whether the use case is score explanation, decision certification, or exact distribution emulation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "baseline_scores": baseline_scores,
        "baseline_margin": baseline_margin,
        "score_values": score_values,
        "all_margin_values": all_margin_values,
        "best_validation": best_validation,
        "cross_objective_overlap": cross_objective_overlap,
    }, indent=2))


if __name__ == "__main__":
    main()
