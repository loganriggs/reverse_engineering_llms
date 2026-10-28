#!/usr/bin/env python3
"""Find and red-team the multilinear core inside the tangent-bundle quotient."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_output_conditioned_suffix_atlas import pca_basis
from run_bilin18_layer17_backward_mlp_native_channels import factor_source, fidelity
from run_bilin18_layer17_backward_mlp_symmetric_tensor import collect_sites
from run_bilin18_layer17_tangent_bundle_quotient import (
    build_role_tensor,
    effect_from_bases,
    enrich_sites,
    tensor_effect,
    slice_sites,
)
from run_bilin18_layer17_tangent_bundle_split_redteam import (
    projector_overlap,
    split_indices,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_tangent_bundle_core_stability.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_core_stability.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_CORE_STABILITY.md")
TENSOR = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_core_stability.pt")


def unfold_modes(tensor):
    modes, spectra = {}, {}
    arrangements = {
        "output": tensor,
        "tangent": tensor.permute(1, 0, 2),
        "context": tensor.permute(2, 0, 1),
    }
    for role, values in arrangements.items():
        u, singular, _ = torch.linalg.svd(values.flatten(1).float(), full_matrices=False)
        energy = singular.double().square()
        shares = energy / energy.sum().clamp_min(1e-30)
        rank90 = int(torch.searchsorted(
            shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype)).item() + 1)
        modes[role] = u
        spectra[role] = {
            "rank90": rank90,
            "top_share": float(shares[0]),
            "shares": shares.tolist(),
        }
    return modes, spectra


def contract_tensor(block, context, tangent, output):
    tensor, *_ = build_role_tensor(block, context, tangent, output)
    return tensor


def score_bases(block, sites, bases, components, batch_size):
    context, tangent, output = bases
    rows = {}
    for component in components:
        target = torch.einsum(
            "bd,bnd->bn", sites["adjoint"][component].float(), sites["output"].float())
        prediction = effect_from_bases(
            block, sites, component, context, tangent, output, batch_size)
        rows[component] = fidelity(target, prediction)
    return rows


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    weights16 = factors["layer16_weights"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    block = model.blocks[17]
    spec = config["discovery"]
    data = factor_source(factors, spec)
    discovery = enrich_sites(
        block, collect_sites(model, spec, data, config), data, spec, weights16,
        config["evaluation_batch_size"])
    external = {}
    for corpus, corpus_spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        corpus_data = factor_source(factors, corpus_spec)
        external[corpus] = enrich_sites(
            block, collect_sites(model, corpus_spec, corpus_data, config),
            corpus_data, corpus_spec, weights16, config["evaluation_batch_size"])

    generator = torch.Generator().manual_seed(config["seed"])
    rows, physical_modes = {}, {}
    saved = {}
    maximum_tensor_error = 0.0
    for split in config["splits"]:
        fit_indices, validation_indices = split_indices(split, len(discovery["state"]), generator)
        fit = slice_sites(discovery, fit_indices)
        validation = slice_sites(discovery, validation_indices)
        base_rank = config["base_rank"]
        context = pca_basis(fit["state"], base_rank)
        tangent = pca_basis(fit["tangent"].flatten(0, 1), base_rank)
        output = pca_basis(fit["output"].flatten(0, 1), base_rank)
        tensor = contract_tensor(block, context, tangent, output)
        modes, spectra = unfold_modes(tensor)
        physical = {
            "context": context @ modes["context"],
            "tangent": tangent @ modes["tangent"],
            "output": output @ modes["output"],
        }
        physical_modes[split] = physical
        if split == "head":
            saved = {"tensor": tensor, **physical}
        evaluation = {}
        for corpus, sites in {"validation": validation, **external}.items():
            evaluation[corpus] = {}
            for rank in config["core_ranks"]:
                bases = (
                    physical["context"][:, :rank],
                    physical["tangent"][:, :rank],
                    physical["output"][:, :rank],
                )
                evaluation[corpus][str(rank)] = score_bases(
                    block, sites, bases, config["components"],
                    config["evaluation_batch_size"])
            rank = 32
            bases = (
                physical["context"][:, :rank],
                physical["tangent"][:, :rank],
                physical["output"][:, :rank],
            )
            reduced_tensor = contract_tensor(block, *bases)
            exact_errors = {}
            for component in config["components"]:
                direct = effect_from_bases(
                    block, sites, component, *bases,
                    config["evaluation_batch_size"])
                contracted = tensor_effect(
                    sites, component, *bases, reduced_tensor,
                    batch_size=config["evaluation_batch_size"])
                error = float(
                    (direct.double() - contracted.double()).norm()
                    / direct.double().norm().clamp_min(1e-30))
                maximum_tensor_error = max(maximum_tensor_error, error)
                exact_errors[component] = error
            evaluation[corpus]["tensor32_exact_error"] = exact_errors
        rows[split] = {"spectra": spectra, "evaluation": evaluation}

    reference = physical_modes["head"]
    overlaps = {
        split: {
            role: {
                str(rank): projector_overlap(
                    reference[role][:, :rank], physical_modes[split][role][:, :rank])
                for rank in (8, 16, 32, 64)
            } for role in ("output", "tangent", "context")
        } for split in config["splits"]
    }
    eval_names = tuple(config["evaluation"])
    ranges32 = {
        corpus: {
            component: max(
                rows[split]["evaluation"][corpus]["32"][component]
                for split in config["splits"])
            - min(
                rows[split]["evaluation"][corpus]["32"][component]
                for split in config["splits"])
            for component in config["components"]
        } for corpus in eval_names
    }
    hypotheses = {
        "C0_role_tensor_has_small_multilinear_rank": all(
            rows[split]["spectra"][role]["rank90"] <= 32
            for split in config["splits"] for role in ("output", "tangent", "context")),
        "C1_top16_physical_core_is_stable": all(
            overlaps[split][role]["16"] >= .70 for split in config["splits"]
            for role in ("output", "tangent", "context")),
        "C2_tucker16_recovers_score_paths": all(
            rows[split]["evaluation"][corpus]["16"][component] >= .80
            for split in config["splits"] for corpus in eval_names
            for component in ("prediction", "target")),
        "C3_tucker32_is_score_sufficient": all(
            rows[split]["evaluation"][corpus]["32"][component] >= .90
            for split in config["splits"] for corpus in eval_names
            for component in ("prediction", "target")),
        "C4_tucker32_recovers_margin": all(
            rows[split]["evaluation"][corpus]["32"]["margin"] >= .60
            for split in config["splits"] for corpus in eval_names),
        "C5_tucker32_is_split_stable": all(
            ranges32[corpus][component] <= .08 for corpus in eval_names
            for component in config["components"]),
        "C6_tucker_tensor_execution_is_exact": maximum_tensor_error <= 1e-5,
        "C7_tensor_axes_are_not_claimed_canonical": True,
    }
    torch.save(saved, TENSOR)
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "maximum_tensor_error": maximum_tensor_error,
        "physical_core_overlap": overlaps,
        "rank32_external_ranges": ranges32,
        "split_results": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Multilinear core stability inside the tangent-bundle quotient", "",
        "Each discovery split produces a rank-64 output × tangent × context tensor. Separate HOSVD unfoldings define physical subspaces for the three tensor roles. The comparison is between subspaces, not individual sign/permutation-dependent axes.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", f"Maximum Tucker-32 tensor/native-CP error: `{maximum_tensor_error:.2e}`.",
              "", "## Spectra and physical-core stability", "",
              "| Split | R90 output/tangent/context | top-16 overlap output/tangent/context |", "|---|---:|---:|"]
    for split in config["splits"]:
        spectra = rows[split]["spectra"]
        lines.append(
            f"| {split} | " + "/".join(str(spectra[role]["rank90"]) for role in ("output", "tangent", "context"))
            + " | " + "/".join(f"{overlaps[split][role]['16']:.3f}" for role in ("output", "tangent", "context")) + " |")
    lines += ["", "## External Tucker fidelity", "",
              "Each cell is prediction/target/margin for Tucker ranks 16 and 32.", "",
              "| Split | Corpus | rank 16 | rank 32 |", "|---|---|---:|---:|"]
    for split in config["splits"]:
        for corpus in eval_names:
            def cell(rank):
                values = rows[split]["evaluation"][corpus][str(rank)]
                return "/".join(f"{values[x]:.3f}" for x in config["components"])
            lines.append(f"| {split} | {corpus} | {cell(16)} | {cell(32)} |")
    lines += [
        "", "A stable physical core would make the rank-64 basis instability a gauge/slack issue. If behavior remains stable but physical core overlap fails, canonical coordinates remain unidentified even though a compact function class exists.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "maximum_tensor_error": maximum_tensor_error,
        "overlap16": {split: {role: overlaps[split][role]["16"] for role in overlaps[split]} for split in overlaps},
        "ranges32": ranges32,
    }, indent=2))


if __name__ == "__main__":
    main()
