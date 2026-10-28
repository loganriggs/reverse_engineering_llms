#!/usr/bin/env python3
"""Test stable Fisher eigenspaces where greedy sparse supports jump."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_router_causal_validation import repeat_batch, suffix_from_layer
from run_bilin18_layer8_suffix_attention_subset import prepare_site, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_canonical_subspace_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_canonical_subspace_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_CANONICAL_SUBSPACE_REDTEAM.md"


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "min": None, "q10": None, "median": None, "max": None}
    return {
        "count": int(len(values)), "min": float(values.min()),
        "q10": float(np.quantile(values, .1)), "median": float(np.median(values)),
        "max": float(values.max()),
    }


def response(model, site, position, mask):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
    )
    return logits, output_fisher_response(logits, tangents)


def eigensystem(response_matrix, amplitudes, floor_fraction):
    absolute = amplitudes.abs()
    floor = absolute.median() * floor_fraction
    valid = absolute >= floor
    unit_response = response_matrix[:, valid] / amplitudes[valid][None]
    gram = unit_response.T @ unit_response
    values, vectors = torch.linalg.eigh(gram)
    order = values.argsort(descending=True)
    return values[order], vectors[:, order], valid, gram


@torch.inference_mode()
def finite_deltas(model, site, position, deltas, scale):
    names = list(deltas)
    x = site["normalized"][0, position].float()
    variants = site["normalized"].repeat(len(names) + 1, 1, 1)
    raw_cosines = {}
    for number, name in enumerate(names, start=1):
        raw = deltas[name]
        delta = raw * (scale * x.norm() / raw.norm().clamp_min(1e-30))
        variants[number, position] = (x + delta).to(variants.dtype)
    block = model.blocks[8]
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(site["pre"], len(variants)) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(site["x0"], len(variants)),
        repeat_batch(site["value_bus"], len(variants)), outputs, position,
    )
    native = logits[0]
    return native, {name: logits[number] - native
                    for number, name in enumerate(names, start=1)}


@torch.inference_mode()
def audit_mask(model, site, position, mask, rank, floor_fraction, edit_fraction):
    native_logits, empty = response(model, site, position, ())
    single = {layer: response(model, site, position, (layer,))[1] for layer in mask}
    pair = {pair_mask: response(model, site, position, pair_mask)[1]
            for pair_mask in itertools.combinations(mask, 2)}
    exact_logits, exact = response(model, site, position, mask)
    if float((exact_logits - native_logits).abs().max()) > 1e-6:
        raise AssertionError("masked analytic passes changed native logits")
    i, j, k = mask
    reconstructed = (pair[(i, j)] + pair[(i, k)] + pair[(j, k)]
                     - single[i] - single[j] - single[k] + empty)
    exact_values, exact_vectors, valid, exact_gram = eigensystem(
        exact, site["amplitudes"], floor_fraction,
    )
    recon_values, recon_vectors, recon_valid, recon_gram = eigensystem(
        reconstructed, site["amplitudes"], floor_fraction,
    )
    if not torch.equal(valid, recon_valid):
        raise AssertionError("amplitude validity mask changed")
    exact_sub = exact_vectors[:, :rank]
    recon_sub = recon_vectors[:, :rank]
    singular = torch.linalg.svdvals(exact_sub.T @ recon_sub)
    projector_overlap = float(singular.square().mean())
    leading_cosine = float((exact_vectors[:, 0] @ recon_vectors[:, 0]).abs())
    if float(exact_vectors[:, 0] @ recon_vectors[:, 0]) < 0:
        recon_vectors[:, 0].mul_(-1)

    valid_directions = site["mode_directions"][:, valid]
    exact_model_sub = valid_directions @ exact_sub
    recon_model_sub = valid_directions @ recon_sub
    x = site["x"]
    deltas = {
        "exact_subspace": -(x @ exact_model_sub) @ exact_model_sub.T,
        "reconstructed_subspace": -(x @ recon_model_sub) @ recon_model_sub.T,
        "exact_leading": valid_directions @ exact_vectors[:, 0],
        "reconstructed_leading": valid_directions @ recon_vectors[:, 0],
    }
    native, effects = finite_deltas(
        model, site, position, deltas, edit_fraction,
    )
    sub_fidelity, sub_cosine, sub_energy = fisher_fidelity(
        native, effects["exact_subspace"], effects["reconstructed_subspace"],
    )
    lead_fidelity, lead_effect_cosine, lead_energy = fisher_fidelity(
        native, effects["exact_leading"], effects["reconstructed_leading"],
    )
    return {
        "mask": list(mask), "valid_modes": int(valid.sum()),
        "gram_relative_error": float((exact_gram - recon_gram).norm()
                                     / exact_gram.norm().clamp_min(1e-30)),
        "projector_overlap": projector_overlap,
        "minimum_principal_cosine": float(singular.min()),
        "leading_vector_cosine": leading_cosine,
        "exact_top_eigengap": float(exact_values[0]
                                    / exact_values[1].clamp_min(1e-30)),
        "exact_rank_boundary_gap": float(exact_values[rank - 1]
                                         / exact_values[rank].clamp_min(1e-30)),
        "subspace_finite_fidelity": sub_fidelity,
        "subspace_finite_cosine": sub_cosine,
        "subspace_target_energy": sub_energy,
        "leading_finite_fidelity": lead_fidelity,
        "leading_finite_cosine": lead_effect_cosine,
        "leading_target_energy": lead_energy,
    }


def aggregate(records, kind):
    rows = [record[kind] for record in records]
    result = {key: summary([row[key] for row in rows]) for key in (
        "valid_modes", "gram_relative_error", "projector_overlap",
        "minimum_principal_cosine", "leading_vector_cosine",
        "exact_top_eigengap", "exact_rank_boundary_gap",
        "subspace_finite_fidelity", "leading_finite_fidelity",
    )}
    identified = [row for row in rows if row["exact_top_eigengap"] >= 1.1]
    result["identified_leading"] = {
        "count": len(identified),
        "vector_cosine": summary([row["leading_vector_cosine"] for row in identified]),
        "finite_fidelity": summary([row["leading_finite_fidelity"] for row in identified]),
    }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    scale_config = json.loads(Path(config["parent_config"]).read_text())
    scale_results = json.loads(Path(config["parent_results"]).read_text())
    blind_config = json.loads(Path(scale_config["parent_config"]).read_text())
    mobius_config = json.loads(Path(blind_config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"], "budget": parent["budget"]}

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(
        model, source_config, parent["queries"],
    )
    corpus_rows = {}
    for corpus, corpus_config in blind_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        corpus_rows[corpus] = source[
            corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]
        ]
    records = []
    for source_record in scale_results["records"]:
        site = prepare_site(
            model, corpus_rows[source_record["corpus"]][source_record["row"]],
            source_record["query"], source_config, tensor_basis, directions, vectors,
        )
        record = {key: source_record[key] for key in ("corpus", "query", "row")}
        for kind in ("failure", "control"):
            print(source_record["corpus"], source_record["query"], kind,
                  source_record[kind]["mask"], flush=True)
            record[kind] = audit_mask(
                model, site, source_config["causal_position"],
                tuple(source_record[kind]["mask"]), config["rank"],
                config["amplitude_floor_fraction"], config["edit_norm_fraction"],
            )
        record["sparse_failure_infinitesimal_fidelity"] = source_record[
            "failure"]["infinitesimal_between_program_fidelity"]
        record["sparse_control_infinitesimal_fidelity"] = source_record[
            "control"]["infinitesimal_between_program_fidelity"]
        records.append(record)

    aggregated = {kind: aggregate(records, kind) for kind in ("failure", "control")}
    identified = aggregated["failure"]["identified_leading"]
    sparse_failure = summary([r["sparse_failure_infinitesimal_fidelity"] for r in records])
    sparse_control = summary([r["sparse_control_infinitesimal_fidelity"] for r in records])
    hypotheses = {
        "C0_pairwise_route_recovers_canonical_subspace": {
            "supported": (aggregated["failure"]["projector_overlap"]["median"] >= .95
                          and aggregated["failure"]["projector_overlap"]["min"] >= .80),
        },
        "C1_canonical_subspace_recovers_finite_effect": {
            "supported": (aggregated["failure"]["subspace_finite_fidelity"]["median"] >= .95
                          and aggregated["failure"]["subspace_finite_fidelity"]["q10"] >= .80),
        },
        "C2_leading_direction_is_stable_when_identified": {
            "supported": (identified["count"] > 0
                          and identified["vector_cosine"]["median"] >= .95
                          and identified["finite_fidelity"]["median"] >= .90),
        },
        "C3_sparse_failures_are_extractor_specific": {
            "supported": (sparse_failure["median"] < .80
                          and aggregated["failure"]["subspace_finite_fidelity"]["median"] >= .95
                          and sparse_control["median"] >= .95
                          and aggregated["control"]["subspace_finite_fidelity"]["median"] >= .95),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "summary": aggregated,
        "sparse_infinitesimal": {"failure": sparse_failure, "control": sparse_control},
        "records": records,
    }, indent=2))
    lines = [
        "# Canonical sensitivity subspace red-team", "",
        "The same blind failures and frozen controls are represented by rank-16 projectors of the unit-input Fisher sensitivity operator instead of discontinuous greedy neuron sets.",
        "", "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "", "| Group | Valid modes | Gram rel. error | Rank-16 overlap median/min | Minimum principal cosine median | Rank-boundary gap median | Subspace finite fidelity median/q10 | Leading cosine median | Leading finite fidelity median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ("failure", "control"):
        row = aggregated[kind]
        lines.append(
            f"| {kind} | {row['valid_modes']['median']:.0f} | {row['gram_relative_error']['median']:.4f} | "
            f"{row['projector_overlap']['median']:.3f}/{row['projector_overlap']['min']:.3f} | "
            f"{row['minimum_principal_cosine']['median']:.3f} | {row['exact_rank_boundary_gap']['median']:.3f} | "
            f"{row['subspace_finite_fidelity']['median']:.3f}/{row['subspace_finite_fidelity']['q10']:.3f} | "
            f"{row['leading_vector_cosine']['median']:.3f} | {row['leading_finite_fidelity']['median']:.3f} |"
        )
    lines += [
        "", f"Sparse infinitesimal fidelity median: failures `{sparse_failure['median']:.3f}`, controls `{sparse_control['median']:.3f}`.",
        "",
        "Projector overlap is the primary identification metric: individual eigenvectors are not canonical inside a nearly degenerate eigenspace. These are sensitivity subspaces for declared route Jacobians, not yet semantic modules or sufficient reconstructions of natural computation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
