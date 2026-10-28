#!/usr/bin/env python3
"""Test whether stable route projectors are relevant to the full native suffix."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_canonical_subspace_redteam import (
    eigensystem,
    finite_deltas,
    response,
    summary,
)
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_router_causal_validation import exact_kl
from run_bilin18_layer8_suffix_attention_subset import prepare_site, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_native_subspace_relevance.json")
OUTPUT = TENSOR_ROOT / "layer8_native_subspace_relevance.json"
REPORT = TENSOR_ROOT / "LAYER8_NATIVE_SUBSPACE_RELEVANCE.md"


@torch.inference_mode()
def audit_mask(model, site, position, mask, rank, floor_fraction, edit_fraction,
               random_count, seed):
    native_logits, empty = response(model, site, position, ())
    single = {layer: response(model, site, position, (layer,))[1] for layer in mask}
    pair = {pair_mask: response(model, site, position, pair_mask)[1]
            for pair_mask in itertools.combinations(mask, 2)}
    route_logits, route = response(model, site, position, mask)
    i, j, k = mask
    reconstructed = (pair[(i, j)] + pair[(i, k)] + pair[(j, k)]
                     - single[i] - single[j] - single[k] + empty)
    # Retain the pairwise check here as an implementation identity, not a new
    # outcome: this experiment's independent target is the full suffix.
    route_values, route_vectors, valid, _ = eigensystem(
        route, site["amplitudes"], floor_fraction,
    )
    recon_values, recon_vectors, recon_valid, _ = eigensystem(
        reconstructed, site["amplitudes"], floor_fraction,
    )
    if not torch.equal(valid, recon_valid):
        raise AssertionError("valid mode mask changed")

    full_logits, full_tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=True,
    )
    if max(float((route_logits - native_logits).abs().max()),
           float((full_logits - native_logits).abs().max())) > 1e-6:
        raise AssertionError("analytic route changed native logits")
    full = output_fisher_response(full_logits, full_tangents)
    full_values, full_vectors, full_valid, _ = eigensystem(
        full, site["amplitudes"], floor_fraction,
    )
    if not torch.equal(valid, full_valid):
        raise AssertionError("full valid mode mask changed")

    route_sub = route_vectors[:, :rank]
    recon_sub = recon_vectors[:, :rank]
    full_sub = full_vectors[:, :rank]
    route_full_singular = torch.linalg.svdvals(route_sub.T @ full_sub)
    route_recon_singular = torch.linalg.svdvals(route_sub.T @ recon_sub)
    model_basis = site["mode_directions"][:, valid]
    route_model = model_basis @ route_sub
    full_model = model_basis @ full_sub
    x = site["x"]
    deltas = {
        "route": -(x @ route_model) @ route_model.T,
        "full": -(x @ full_model) @ full_model.T,
    }
    generator = torch.Generator(device=x.device).manual_seed(seed)
    for number in range(random_count):
        raw = torch.randn(
            model_basis.shape[1], rank, generator=generator, device=x.device,
        )
        random_coefficients, _ = torch.linalg.qr(raw, mode="reduced")
        random_model = model_basis @ random_coefficients
        deltas[f"random_{number}"] = -(x @ random_model) @ random_model.T
    native, effects = finite_deltas(model, site, position, deltas, edit_fraction)
    fidelity, cosine, energy = fisher_fidelity(native, effects["full"], effects["route"])
    route_kl = exact_kl(native, native + effects["route"])
    full_kl = exact_kl(native, native + effects["full"])
    random_kl = [exact_kl(native, native + effects[f"random_{number}"])
                 for number in range(random_count)]
    random_median = float(np.median(random_kl))
    return {
        "mask": list(mask), "valid_modes": int(valid.sum()),
        "route_reconstruction_projector_overlap": float(
            route_recon_singular.square().mean()),
        "route_full_projector_overlap": float(route_full_singular.square().mean()),
        "route_full_minimum_principal_cosine": float(route_full_singular.min()),
        "route_rank_boundary_gap": float(route_values[rank - 1]
                                         / route_values[rank].clamp_min(1e-30)),
        "full_rank_boundary_gap": float(full_values[rank - 1]
                                        / full_values[rank].clamp_min(1e-30)),
        "route_full_finite_fidelity": fidelity,
        "route_full_finite_cosine": cosine,
        "full_effect_energy": energy,
        "route_kl": route_kl, "full_kl": full_kl,
        "random_kl_median": random_median,
        "random_kl_q90": float(np.quantile(random_kl, .9)),
        "route_to_random_kl": route_kl / max(random_median, 1e-30),
        "full_to_random_kl": full_kl / max(random_median, 1e-30),
    }


def aggregate(records, kind):
    rows = [record[kind] for record in records]
    return {key: summary([row[key] for row in rows]) for key in (
        "valid_modes", "route_reconstruction_projector_overlap",
        "route_full_projector_overlap", "route_full_minimum_principal_cosine",
        "route_rank_boundary_gap", "full_rank_boundary_gap",
        "route_full_finite_fidelity", "route_kl", "full_kl",
        "random_kl_median", "route_to_random_kl", "full_to_random_kl",
    )}


def main():
    config = json.loads(CONFIG.read_text())
    canonical_config = json.loads(Path(config["parent_config"]).read_text())
    canonical_results = json.loads(Path(config["parent_results"]).read_text())
    scale_config = json.loads(Path(canonical_config["parent_config"]).read_text())
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
    for number, source_record in enumerate(canonical_results["records"]):
        site = prepare_site(
            model, corpus_rows[source_record["corpus"]][source_record["row"]],
            source_record["query"], source_config, tensor_basis, directions, vectors,
        )
        record = {key: source_record[key] for key in ("corpus", "query", "row")}
        for kind_number, kind in enumerate(("failure", "control")):
            print(source_record["corpus"], source_record["query"], kind, flush=True)
            record[kind] = audit_mask(
                model, site, source_config["causal_position"],
                tuple(source_record[kind]["mask"]), config["rank"],
                config["amplitude_floor_fraction"], config["edit_norm_fraction"],
                config["random_subspaces"],
                config["seed"] + 1000 * number + kind_number,
            )
        records.append(record)
    aggregated = {kind: aggregate(records, kind) for kind in ("failure", "control")}
    n0 = all(aggregated[kind]["route_full_projector_overlap"]["median"] >= .80
             and aggregated[kind]["route_full_projector_overlap"]["q10"] >= .50
             for kind in aggregated)
    n1 = all(aggregated[kind]["route_full_finite_fidelity"]["median"] >= .80
             and aggregated[kind]["route_full_finite_fidelity"]["q10"] >= .50
             for kind in aggregated)
    n2 = all(aggregated[kind]["full_to_random_kl"]["median"] >= 3
             for kind in aggregated)
    n3 = all(aggregated[kind]["route_to_random_kl"]["median"] >= 2
             for kind in aggregated)
    hypotheses = {
        "N0_route_subspace_matches_full_native_subspace": {"supported": n0},
        "N1_route_subspace_matches_full_finite_effect": {"supported": n1},
        "N2_full_native_subspace_beats_random": {"supported": n2},
        "N3_route_subspace_beats_random": {"supported": n3},
        "N4_stable_route_object_is_native_relevant": {"supported": n0 and n1},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses,
        "summary": aggregated, "records": records,
    }, indent=2))
    lines = [
        "# Native relevance of canonical route subspaces", "",
        "Stable three-layer route projectors are compared with the complete suffix-Jacobian projector and with seeded random rank-16 subspaces on the frozen blind examples.",
        "", "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "", "| Group | Route/full projector overlap median/q10 | Min principal cosine median | Route/full finite fidelity median/q10 | Route KL | Full KL | Random KL | Route/random | Full/random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ("failure", "control"):
        row = aggregated[kind]
        lines.append(
            f"| {kind} | {row['route_full_projector_overlap']['median']:.3f}/{row['route_full_projector_overlap']['q10']:.3f} | "
            f"{row['route_full_minimum_principal_cosine']['median']:.3f} | "
            f"{row['route_full_finite_fidelity']['median']:.3f}/{row['route_full_finite_fidelity']['q10']:.3f} | "
            f"{row['route_kl']['median']:.3e} | {row['full_kl']['median']:.3e} | {row['random_kl_median']['median']:.3e} | "
            f"{row['route_to_random_kl']['median']:.1f}x | {row['full_to_random_kl']['median']:.1f}x |"
        )
    lines += [
        "",
        "The full projector is a local causal-sensitivity object, not a proof of natural feature use. All finite edits are norm-matched; literal natural-component removal and semantic transfer remain separate tests.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
