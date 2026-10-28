#!/usr/bin/env python3
"""Red-team tangent-bundle quotient stability across discovery splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_output_conditioned_suffix_atlas import pca_basis
from run_bilin18_layer17_backward_mlp_native_channels import factor_source
from run_bilin18_layer17_backward_mlp_symmetric_tensor import collect_sites
from run_bilin18_layer17_tangent_bundle_quotient import (
    effect_from_bases,
    enrich_sites,
    scaled_adjoint_basis,
    slice_sites,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_layer17_backward_mlp_native_channels import fidelity


CONFIG = Path("configs/bilin18_layer17_tangent_bundle_split_redteam.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_split_redteam.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_SPLIT_REDTEAM.md")


def split_indices(name, count, generator):
    all_indices = torch.arange(count)
    if name == "head":
        fit = all_indices[:192]
    elif name == "tail":
        fit = all_indices[64:]
    elif name == "interleaved":
        fit = all_indices[all_indices.remainder(4) != 0]
    elif name == "random":
        fit = torch.randperm(count, generator=generator)[:192].sort().values
    else:
        raise ValueError(name)
    mask = torch.ones(count, dtype=torch.bool)
    mask[fit] = False
    return fit, all_indices[mask]


def projector_overlap(first, second):
    return float((first.double().T @ second.double()).square().sum() / first.shape[1])


def mean_endpoint(row):
    return float(np.mean([row[name] for name in ("prediction", "target", "margin")]))


def spearman(first, second):
    """Spearman correlation for the tie-free four-split diagnostic."""
    first_rank = np.argsort(np.argsort(np.asarray(first))).astype(np.float64)
    second_rank = np.argsort(np.argsort(np.asarray(second))).astype(np.float64)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    return float(
        (first_rank * second_rank).sum()
        / np.sqrt((np.square(first_rank).sum() * np.square(second_rank).sum())))


@torch.inference_mode()
def evaluate(block, sites, bases, components, batch_size):
    context, tangent, output, adjoint = bases
    rows = {"role_separated": {}, "shared_state_pca": {}, "swapped_roles": {}}
    for component in components:
        target = torch.einsum(
            "bd,bnd->bn", sites["adjoint"][component].float(), sites["output"].float())
        predictions = {
            "role_separated": effect_from_bases(
                block, sites, component, context, tangent, output, batch_size),
            "shared_state_pca": effect_from_bases(
                block, sites, component, context, context, adjoint, batch_size),
            "swapped_roles": effect_from_bases(
                block, sites, component, tangent, context, output, batch_size),
        }
        for name, prediction in predictions.items():
            rows[name][component] = fidelity(target, prediction)
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

    discovery_spec = config["discovery"]
    discovery_factor = factor_source(factors, discovery_spec)
    discovery = enrich_sites(
        block,
        collect_sites(model, discovery_spec, discovery_factor, config),
        discovery_factor, discovery_spec, weights16, config["evaluation_batch_size"])
    external = {}
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data = factor_source(factors, spec)
        external[corpus] = enrich_sites(
            block, collect_sites(model, spec, data, config),
            data, spec, weights16, config["evaluation_batch_size"])

    generator = torch.Generator().manual_seed(config["seed"])
    split_rows, basis_rows = {}, {}
    reference_bases = None
    for name in config["splits"]:
        fit_indices, validation_indices = split_indices(name, len(discovery["state"]), generator)
        fit = slice_sites(discovery, fit_indices)
        validation = slice_sites(discovery, validation_indices)
        rank = config["rank"]
        bases = (
            pca_basis(fit["state"], rank),
            pca_basis(fit["tangent"].flatten(0, 1), rank),
            pca_basis(fit["output"].flatten(0, 1), rank),
            scaled_adjoint_basis(fit, config["components"], rank),
        )
        if reference_bases is None:
            reference_bases = bases
        basis_rows[name] = {
            role: projector_overlap(reference_bases[index], bases[index])
            for index, role in enumerate(("context", "tangent", "output", "adjoint"))
        }
        split_rows[name] = {
            "fit_indices": fit_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "validation": evaluate(
                block, validation, bases, config["components"],
                config["evaluation_batch_size"]),
            "evaluation": {
                corpus: evaluate(
                    block, sites, bases, config["components"],
                    config["evaluation_batch_size"])
                for corpus, sites in external.items()
            },
        }

    eval_names = tuple(config["evaluation"])
    ranges = {
        corpus: {
            component: max(
                split_rows[name]["evaluation"][corpus]["role_separated"][component]
                for name in config["splits"])
            - min(
                split_rows[name]["evaluation"][corpus]["role_separated"][component]
                for name in config["splits"])
            for component in config["components"]
        } for corpus in eval_names
    }
    validation_means = [
        mean_endpoint(split_rows[name]["validation"]["role_separated"])
        for name in config["splits"]
    ]
    fineweb_means = [
        mean_endpoint(split_rows[name]["evaluation"]["fineweb"]["role_separated"])
        for name in config["splits"]
    ]
    validation_external_spearman = spearman(validation_means, fineweb_means)
    hypotheses = {
        "R0_all_split_bases_are_discovery_only": all(
            len(row["fit_indices"]) == 192 and len(row["validation_indices"]) == 64
            and not set(row["fit_indices"]) & set(row["validation_indices"])
            for row in split_rows.values()),
        "R1_score_fidelity_is_split_stable": all(
            ranges[corpus][component] <= .05 for corpus in eval_names
            for component in ("prediction", "target")),
        "R2_margin_fidelity_is_split_stable": all(
            ranges[corpus]["margin"] <= .10 for corpus in eval_names),
        "R3_every_split_recovers_scores": all(
            split_rows[name]["evaluation"][corpus]["role_separated"][component] >= .80
            for name in config["splits"] for corpus in eval_names
            for component in ("prediction", "target")),
        "R4_correct_roles_beat_swapped_roles": all(
            mean_endpoint(split_rows[name]["evaluation"][corpus]["role_separated"])
            - mean_endpoint(split_rows[name]["evaluation"][corpus]["swapped_roles"]) >= .10
            for name in config["splits"] for corpus in eval_names),
        "R5_role_bases_are_subspace_stable": all(
            basis_rows[name][role] >= .75 for name in config["splits"]
            for role in ("context", "tangent", "output")),
        "R6_margin_is_systematically_harder": all(
            np.mean([
                split_rows[name]["evaluation"][corpus]["role_separated"][component]
                for component in ("prediction", "target")])
            - split_rows[name]["evaluation"][corpus]["role_separated"]["margin"] >= .15
            for name in config["splits"] for corpus in eval_names),
        "R7_heldout_split_predicts_external_behavior": validation_external_spearman > 0,
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "external_ranges": ranges,
        "basis_overlap_with_head": basis_rows,
        "validation_fineweb_spearman": validation_external_spearman,
        "split_results": split_rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Split-stability red team of the tangent-bundle quotient", "",
        "Four rank-64 basis triples are fit to different 192-of-256 discovery-site subsets. Each is tested on its complementary 64 sites and on the same disjoint external corpora. The swapped-role quotient has exactly the same ranks and learned subspaces as the positive condition.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += ["", "## External role-separated fidelity", "",
              "Each cell is prediction/target/margin.", "",
              "| Split | FineWeb | WikiText | TinyStories | Validation |", "|---|---:|---:|---:|---:|"]
    for name in config["splits"]:
        row = split_rows[name]
        def cell(values):
            return "/".join(f"{values[x]:.3f}" for x in config["components"])
        lines.append(
            f"| {name} | {cell(row['evaluation']['fineweb']['role_separated'])} | "
            f"{cell(row['evaluation']['wikitext']['role_separated'])} | "
            f"{cell(row['evaluation']['tinystories']['role_separated'])} | "
            f"{cell(row['validation']['role_separated'])} |")
    lines += ["", "## Basis stability and matched role control", "",
              "| Split | context/tangent/output overlap with head | Mean role − swapped: FW/Wiki/Tiny |", "|---|---:|---:|"]
    for name in config["splits"]:
        overlap = basis_rows[name]
        gaps = []
        for corpus in eval_names:
            gaps.append(
                mean_endpoint(split_rows[name]["evaluation"][corpus]["role_separated"])
                - mean_endpoint(split_rows[name]["evaluation"][corpus]["swapped_roles"]))
        lines.append(
            f"| {name} | {overlap['context']:.3f}/{overlap['tangent']:.3f}/{overlap['output']:.3f} | "
            + "/".join(f"{gap:.3f}" for gap in gaps) + " |")
    lines += [
        "", f"Validation-to-external-FineWeb Spearman across four splits: `{validation_external_spearman:.3f}`.",
        "", "Stable external behavior with imperfect basis overlap would mean that the functional quotient is more identifiable than its individual PCA axes. An unstable margin with stable score paths would reinforce precision-conditioned support rather than refute the score-level quotient.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "ranges": ranges,
        "basis_overlap": basis_rows,
        "validation_fineweb_spearman": validation_external_spearman,
    }, indent=2))


if __name__ == "__main__":
    main()
