#!/usr/bin/env python3
"""Separate stable sensitivity operators from their state-conditioned action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_canonical_subspace_redteam import eigensystem, finite_deltas, response, summary
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_router_causal_validation import exact_kl, repeat_batch, suffix_from_layer
from run_bilin18_layer8_suffix_attention_subset import prepare_site, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_state_contraction_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_state_contraction_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_STATE_CONTRACTION_REDTEAM.md"


@torch.inference_mode()
def finite_literal(model, site, position, deltas):
    names = list(deltas)
    variants = site["normalized"].repeat(len(names) + 1, 1, 1)
    x = site["normalized"][0, position].float()
    for number, name in enumerate(names, start=1):
        variants[number, position] = (x + deltas[name]).to(variants.dtype)
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


def cosine(left, right):
    return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = .5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left, right):
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


@torch.inference_mode()
def audit(model, site, position, mask, rank, floor_fraction, scales, random_count, seed):
    route_logits, route = response(model, site, position, mask)
    full_logits, full_tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=True,
    )
    if float((route_logits - full_logits).abs().max()) > 1e-6:
        raise AssertionError("analytic passes changed native logits")
    full = output_fisher_response(full_logits, full_tangents)
    route_values, route_vectors, valid, _ = eigensystem(
        route, site["amplitudes"], floor_fraction,
    )
    full_values, full_vectors, full_valid, _ = eigensystem(
        full, site["amplitudes"], floor_fraction,
    )
    if not torch.equal(valid, full_valid):
        raise AssertionError("valid mode masks differ")
    route_sub, full_sub = route_vectors[:, :rank], full_vectors[:, :rank]
    if float(route_vectors[:, 0] @ full_vectors[:, 0]) < 0:
        route_vectors[:, 0].mul_(-1)
    basis = site["mode_directions"][:, valid]
    route_model, full_model = basis @ route_sub, basis @ full_sub
    route_leading = basis @ route_vectors[:, 0]
    full_leading = basis @ full_vectors[:, 0]
    x = site["x"]
    route_projection = (x @ route_model) @ route_model.T
    full_projection = (x @ full_model) @ full_model.T
    projection_cosine = cosine(route_projection, full_projection)
    leading_cosine = cosine(route_leading, full_leading)

    normalized = {}
    for scale in scales:
        native, effects = finite_deltas(model, site, position, {
            "route_projection": -route_projection,
            "full_projection": -full_projection,
            "route_leading": route_leading,
            "full_leading": full_leading,
        }, scale)
        projection_fidelity, projection_effect_cosine, _ = fisher_fidelity(
            native, effects["full_projection"], effects["route_projection"],
        )
        leading_fidelity, leading_effect_cosine, _ = fisher_fidelity(
            native, effects["full_leading"], effects["route_leading"],
        )
        normalized[str(scale)] = {
            "projection_fidelity": projection_fidelity,
            "projection_effect_cosine": projection_effect_cosine,
            "leading_fidelity": leading_fidelity,
            "leading_effect_cosine": leading_effect_cosine,
        }

    generator = torch.Generator(device=x.device).manual_seed(seed)
    literal_deltas = {"route": -route_projection, "full": -full_projection}
    for number in range(random_count):
        raw = torch.randn(basis.shape[1], rank, generator=generator, device=x.device)
        coefficients, _ = torch.linalg.qr(raw, mode="reduced")
        random_model = basis @ coefficients
        literal_deltas[f"random_{number}"] = -(x @ random_model) @ random_model.T
    native, literal_effects = finite_literal(model, site, position, literal_deltas)
    literal_fidelity, literal_cosine, _ = fisher_fidelity(
        native, literal_effects["full"], literal_effects["route"],
    )
    route_kl = exact_kl(native, native + literal_effects["route"])
    full_kl = exact_kl(native, native + literal_effects["full"])
    random_kl = [exact_kl(native, native + literal_effects[f"random_{number}"])
                 for number in range(random_count)]
    random_median = float(np.median(random_kl))
    return {
        "mask": list(mask), "leading_input_cosine": leading_cosine,
        "projection_input_cosine": projection_cosine,
        "route_projection_norm_fraction": float(route_projection.norm() / x.norm()),
        "full_projection_norm_fraction": float(full_projection.norm() / x.norm()),
        "projection_norm_ratio": float(route_projection.norm()
                                       / full_projection.norm().clamp_min(1e-30)),
        "literal_fidelity": literal_fidelity,
        "literal_effect_cosine": literal_cosine,
        "route_literal_kl": route_kl, "full_literal_kl": full_kl,
        "random_literal_kl_median": random_median,
        "full_to_random_literal_kl": full_kl / max(random_median, 1e-30),
        "route_to_random_literal_kl": route_kl / max(random_median, 1e-30),
        "normalized": normalized,
    }


def aggregate(records, kind, scales):
    rows = [record[kind] for record in records]
    result = {key: summary([row[key] for row in rows]) for key in (
        "leading_input_cosine", "projection_input_cosine",
        "route_projection_norm_fraction", "full_projection_norm_fraction",
        "projection_norm_ratio", "literal_fidelity", "route_literal_kl",
        "full_literal_kl", "random_literal_kl_median",
        "full_to_random_literal_kl", "route_to_random_literal_kl",
    )}
    result["normalized"] = {}
    for scale in scales:
        result["normalized"][str(scale)] = {key: summary([
            row["normalized"][str(scale)][key] for row in rows
        ]) for key in ("projection_fidelity", "leading_fidelity")}
    return result


def main():
    config = json.loads(CONFIG.read_text())
    native_config = json.loads(Path(config["parent_config"]).read_text())
    native_results = json.loads(Path(config["parent_results"]).read_text())
    canonical_config = json.loads(Path(native_config["parent_config"]).read_text())
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
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    corpus_rows = {}
    for corpus, corpus_config in blind_config["evaluation"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        corpus_rows[corpus] = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    records = []
    for number, source_record in enumerate(native_results["records"]):
        site = prepare_site(model, corpus_rows[source_record["corpus"]][source_record["row"]],
                            source_record["query"], source_config, tensor_basis, directions, vectors)
        record = {key: source_record[key] for key in ("corpus", "query", "row")}
        for kind_number, kind in enumerate(("failure", "control")):
            print(source_record["corpus"], source_record["query"], kind, flush=True)
            record[kind] = audit(
                model, site, source_config["causal_position"], tuple(source_record[kind]["mask"]),
                config["rank"], config["amplitude_floor_fraction"],
                config["normalized_scales"], config["random_subspaces"],
                config["seed"] + 1000 * number + kind_number,
            )
        records.append(record)
    aggregated = {kind: aggregate(records, kind, config["normalized_scales"])
                  for kind in ("failure", "control")}
    projection_cosines = [record[kind]["projection_input_cosine"]
                          for record in records for kind in ("failure", "control")]
    finite_fidelities = [record[kind]["normalized"]["0.05"]["projection_fidelity"]
                         for record in records for kind in ("failure", "control")]
    correlation = spearman(projection_cosines, finite_fidelities)
    d0 = all(aggregated[k]["leading_input_cosine"]["median"] >= .95
             and aggregated[k]["leading_input_cosine"]["q10"] >= .80
             and aggregated[k]["normalized"]["0.01"]["leading_fidelity"]["median"] >= .95
             and aggregated[k]["normalized"]["0.01"]["leading_fidelity"]["q10"] >= .80
             for k in aggregated)
    d2 = all(aggregated[k]["projection_input_cosine"]["median"] >= .90
             and aggregated[k]["projection_input_cosine"]["q10"] >= .70 for k in aggregated)
    d3 = all(aggregated[k]["literal_fidelity"]["median"] >= .80
             and aggregated[k]["literal_fidelity"]["q10"] >= .50 for k in aggregated)
    d4 = all(aggregated[k]["full_to_random_literal_kl"]["median"] >= 3 for k in aggregated)
    hypotheses = {
        "D0_leading_sensitivity_direction_is_native_relevant": {"supported": d0},
        "D1_state_contraction_predicts_finite_route_error": {"supported": correlation >= .60},
        "D2_route_recovers_full_active_projection": {"supported": d2},
        "D3_literal_projection_effect_transfers": {"supported": d3},
        "D4_full_literal_component_beats_random": {"supported": d4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "projection_fidelity_spearman": correlation,
                                  "summary": aggregated, "records": records}, indent=2))
    lines = ["# State contraction red-team", "",
             "Stable route/full sensitivity projectors are contracted with the current residual state. Leading perturbation directions, active projections, normalized edits, and literal natural-component removals are kept distinct.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Projection-input cosine versus 5%-effect fidelity Spearman: `{correlation:.3f}`.", "",
              "| Group | Leading input cosine med/q10 | Leading 1%-effect fidelity med/q10 | Projection input cosine med/q10 | Projection norm route/full | Projection 5%-effect fidelity med/q10 | Literal effect fidelity med/q10 | Literal KL route/full/random | Full/random literal KL |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("failure", "control"):
        row = aggregated[kind]
        lines.append(
            f"| {kind} | {row['leading_input_cosine']['median']:.3f}/{row['leading_input_cosine']['q10']:.3f} | "
            f"{row['normalized']['0.01']['leading_fidelity']['median']:.3f}/{row['normalized']['0.01']['leading_fidelity']['q10']:.3f} | "
            f"{row['projection_input_cosine']['median']:.3f}/{row['projection_input_cosine']['q10']:.3f} | "
            f"{row['route_projection_norm_fraction']['median']:.3f}/{row['full_projection_norm_fraction']['median']:.3f} | "
            f"{row['normalized']['0.05']['projection_fidelity']['median']:.3f}/{row['normalized']['0.05']['projection_fidelity']['q10']:.3f} | "
            f"{row['literal_fidelity']['median']:.3f}/{row['literal_fidelity']['q10']:.3f} | "
            f"{row['route_literal_kl']['median']:.2e}/{row['full_literal_kl']['median']:.2e}/{row['random_literal_kl_median']['median']:.2e} | "
            f"{row['full_to_random_literal_kl']['median']:.1f}x |"
        )
    lines += ["", "A high-sensitivity direction is a potential local verb. Its contraction with the naturally occurring state determines whether that verb is active. Neither establishes semantic reuse without cross-context transfer."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
