#!/usr/bin/env python3
"""Diagnose ill-conditioned sparse selection in pairwise Mobius responses."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_pair_mobius_reconstruction import response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_program_stability.json")
OUTPUT = TENSOR_ROOT / "layer8_program_stability.json"
REPORT = TENSOR_ROOT / "LAYER8_PROGRAM_STABILITY.md"


def greedy_minimum_gap(gram, amplitudes, budget):
    available = torch.ones(gram.shape[0], dtype=torch.bool, device=gram.device)
    indicator = torch.zeros(gram.shape[0], device=gram.device)
    denominator = torch.zeros((), device=gram.device)
    diagonal = gram.diagonal()
    gaps = []
    for _ in range(budget):
        cross = gram @ indicator
        current = indicator @ cross
        score = (current + 2 * cross + diagonal) / (
            denominator + amplitudes.square()
        ).clamp_min(1e-30)
        score = score.masked_fill(~available, -torch.inf)
        top = torch.topk(score, 2)
        chosen = int(top.indices[0])
        gaps.append(float((top.values[0] - top.values[1]) /
                          top.values[0].abs().clamp_min(1e-30)))
        indicator[chosen] = 1; available[chosen] = False
        denominator = denominator + amplitudes[chosen].square()
    return min(gaps)


def auroc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(np.mean(positive[:, None] > negative[None])
                 + 0.5 * np.mean(positive[:, None] == negative[None]))


def choose_certificate(records):
    margins = np.asarray([row["minimum_gap"] for row in records])
    errors = np.asarray([row["whitened_gram_error"] for row in records])
    good = np.asarray([not row["failure"] for row in records])
    best = None
    for margin in np.quantile(margins, np.linspace(0, 1, 41)):
        for error in np.quantile(errors, np.linspace(0, 1, 41)):
            keep = (margins >= margin) & (errors <= error)
            if not keep.any():
                continue
            precision = float(good[keep].mean()); coverage = float(keep.mean())
            if precision >= .99 and (best is None or (coverage, precision) > (best["coverage"], best["precision"])):
                best = {"margin": float(margin), "error": float(error),
                        "coverage": coverage, "precision": precision}
    return best


def certificate_result(records, certificate):
    keep = np.asarray([row["minimum_gap"] >= certificate["margin"]
                       and row["whitened_gram_error"] <= certificate["error"] for row in records])
    good = np.asarray([not row["failure"] for row in records])
    return {"coverage": float(keep.mean()),
            "precision": float(good[keep].mean()) if keep.any() else float("nan"),
            "kept": int(keep.sum()), "failures_kept": int((keep & ~good).sum())}


def main():
    config = json.loads(CONFIG.read_text())
    mobius_config = json.loads(Path(config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": mobius_config["sites_per_query"], "budget": parent["budget"]}
    frozen = json.loads((TENSOR_ROOT / "layer8_pair_mobius_reconstruction.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    layers = parent["suffix_attention_layers"]
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(parent["evaluation"].items()):
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], mobius_config["sites_per_query"])
        frozen_lookup = {(site["query"], site["row"], tuple(row["mask"])): row
                         for site in frozen["results"][corpus]["records"] for row in site["triples"]}
        records = []
        for number, (query, row_index, raw) in enumerate(rows):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            _, empty = response(model, site, source_config["causal_position"], ())
            single = {layer: response(model, site, source_config["causal_position"], (layer,))[1]
                      for layer in layers}
            pair = {mask: response(model, site, source_config["causal_position"], mask)[1]
                    for mask in itertools.combinations(layers, 2)}
            for triple in itertools.combinations(layers, 3):
                _, exact = response(model, site, source_config["causal_position"], triple)
                i, j, k = triple
                reconstructed = (pair[(i, j)] + pair[(i, k)] + pair[(j, k)]
                                 - single[i] - single[j] - single[k] + empty)
                exact_gram = exact.T @ exact; reconstructed_gram = reconstructed.T @ reconstructed
                amplitude = site["amplitudes"].abs()
                floor = float(amplitude.median()) * 1e-6
                scale = amplitude.clamp_min(max(floor, 1e-30))
                outer = scale[:, None] * scale[None]
                whitened_exact = exact_gram / outer
                whitened_reconstructed = reconstructed_gram / outer
                frozen_row = frozen_lookup[(query, row_index, triple)]
                records.append({
                    "query": query, "row": row_index, "mask": list(triple),
                    "finite_effect_fidelity": frozen_row["finite_effect_fidelity"],
                    "failure": frozen_row["finite_effect_fidelity"] < config["failure_fidelity"],
                    "response_cosine": frozen_row["response_cosine"],
                    "minimum_gap": greedy_minimum_gap(exact_gram, site["amplitudes"], parent["budget"]),
                    "gram_error": float((exact_gram - reconstructed_gram).norm() /
                                        exact_gram.norm().clamp_min(1e-30)),
                    "whitened_gram_error": float(
                        (whitened_exact - whitened_reconstructed).norm()
                        / whitened_exact.norm().clamp_min(1e-30)),
                })
                del exact, reconstructed, exact_gram, reconstructed_gram
            print(corpus, "stability site", number + 1, "of", len(rows), flush=True)
        labels = [row["failure"] for row in records]
        failures = sum(labels)
        failure_sites = len(set((row["query"], row["row"]) for row in records if row["failure"]))
        results[corpus] = {"summary": {
            "triples": len(records), "failures": failures,
            "failure_fraction": failures / len(records),
            "sites": len(rows), "failure_sites": failure_sites,
            "failure_site_fraction": failure_sites / len(rows),
            "auroc_negative_gap": auroc([-row["minimum_gap"] for row in records], labels),
            "auroc_whitened_error": auroc([row["whitened_gram_error"] for row in records], labels),
            "auroc_gram_error": auroc([row["gram_error"] for row in records], labels),
            "auroc_negative_cosine": auroc([-row["response_cosine"] for row in records], labels),
        }, "records": records}
    certificate = choose_certificate(results["fineweb"]["records"])
    certificate_results = {corpus: certificate_result(result["records"], certificate)
                           for corpus, result in results.items()}
    rare = all(result["summary"]["failure_fraction"] < .02
               and result["summary"]["failure_site_fraction"] <= .20 for result in results.values())
    margin = all(result["summary"]["auroc_negative_gap"] >= .80 for result in results.values())
    whitened = all(result["summary"]["auroc_whitened_error"] >= .85 for result in results.values())
    cosine_bad = any(result["summary"]["auroc_negative_cosine"] < .75 for result in results.values())
    certifies = certificate_results["wikitext"]["coverage"] >= .90 and certificate_results["wikitext"]["precision"] >= .99
    hypotheses = {
        "S0_failures_are_rare_and_clustered": {"supported": bool(rare)},
        "S1_low_greedy_margin_predicts_failure": {"supported": bool(margin)},
        "S2_cost_whitened_error_predicts_failure": {"supported": bool(whitened)},
        "S3_global_response_cosine_is_inadequate": {"supported": bool(cosine_bad)},
        "S4_stable_pairwise_regime_is_certifiable": {"supported": bool(certifies)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "certificate": certificate,
                                  "certificate_results": certificate_results,
                                  "results": results}, indent=2))
    lines = ["# Stability boundaries of Layer-8 sparse program extraction", "",
             "Rare failures of second-order response reconstruction are tested as ill-conditioned greedy selection boundaries. The causal failure labels are frozen from exact finite interventions.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Failures/triples | Failure sites | AUROC: -gap | AUROC: whitened error | AUROC: raw Gram error | AUROC: -cosine | Certificate coverage/precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]; cert = certificate_results[corpus]
        lines.append(f"| {corpus} | {row['failures']}/{row['triples']} | {row['failure_sites']}/{row['sites']} | {row['auroc_negative_gap']:.3f} | {row['auroc_whitened_error']:.3f} | {row['auroc_gram_error']:.3f} | {row['auroc_negative_cosine']:.3f} | {cert['coverage']:.3f}/{cert['precision']:.3f} |")
    lines += ["", f"FineWeb certificate: minimum gap `{certificate['margin']:.3e}`, maximum whitened error `{certificate['error']:.3e}`.", "",
              "A useful certificate separates the regular second-order regime from singular states. Failure means these two scalar diagnostics do not safely identify the boundary; it does not make the bulk reconstruction false."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
