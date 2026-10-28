#!/usr/bin/env python3
"""Cross-validated additive and pairwise descriptions of suffix-mask effects."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_suffix_mask_grammar.json")
OUTPUT = TENSOR_ROOT / "layer8_suffix_mask_grammar.json"
REPORT = TENSOR_ROOT / "LAYER8_SUFFIX_MASK_GRAMMAR.md"
FIGURE = TENSOR_ROOT / "layer8_suffix_mask_grammar.png"


def design(masks, order):
    layers = list(range(9, 18))
    bits = np.asarray([[float(layer in mask) for layer in layers] for mask in masks])
    if order == "additive":
        return bits
    pairs = np.asarray([[row[i] * row[j] for i in range(9) for j in range(i + 1, 9)] for row in bits])
    return np.concatenate([bits, pairs], axis=1)


def ridge_fit(x, y, alpha):
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    matrix = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(matrix.shape[1]); penalty[0, 0] = 0
    coefficient = np.linalg.solve(matrix.T @ matrix + alpha * penalty, matrix.T @ y)
    return mean, scale, coefficient


def ridge_predict(model, x):
    mean, scale, coefficient = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coefficient


def choose_alpha(x, y, indices, alphas, folds):
    scores = {alpha: [] for alpha in alphas}
    for fold in range(folds):
        valid = indices[np.arange(len(indices)) % folds == fold]
        train = np.setdiff1d(indices, valid)
        for alpha in alphas:
            prediction = ridge_predict(ridge_fit(x[train], y[train], alpha), x[valid])
            scores[alpha].append(float(np.mean((prediction - y[valid]) ** 2)))
    return min(alphas, key=lambda alpha: (np.mean(scores[alpha]), alpha))


def cross_validate(x, y, config):
    rng = np.random.default_rng(config["seed"])
    permutation = rng.permutation(len(y))
    predictions = np.zeros_like(y, dtype=np.float64)
    regrets = []
    selected_alphas = []
    for fold in range(config["folds"]):
        valid = permutation[np.arange(len(permutation)) % config["folds"] == fold]
        train = np.setdiff1d(np.arange(len(y)), valid)
        alpha = choose_alpha(x, y, train, config["ridge_alphas"], config["inner_folds"])
        selected_alphas.append(alpha)
        prediction = ridge_predict(ridge_fit(x[train], y[train], alpha), x[valid])
        predictions[valid] = prediction
        picked = valid[int(np.argmax(prediction))]
        regrets.append(float(y[valid].max() - y[picked]))
    denominator = np.square(y - y.mean()).sum()
    r2 = 1 - np.square(y - predictions).sum() / max(denominator, 1e-30)
    return {"r2": float(r2), "median_regret": float(np.median(regrets)),
            "max_regret": float(max(regrets)), "regrets": regrets,
            "alphas": selected_alphas}


def aggregate(records):
    result = {}
    for target in ("full_fidelity", "objective_efficiency"):
        result[target] = {}
        for order in ("additive", "pairwise"):
            rows = [record[target][order] for record in records]
            result[target][order] = {
                "median_r2": float(np.median([row["r2"] for row in rows])),
                "q10_r2": float(np.quantile([row["r2"] for row in rows], .10)),
                "median_regret": float(np.median([row["median_regret"] for row in rows])),
                "q90_regret": float(np.quantile([row["median_regret"] for row in rows], .90)),
            }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    results = {}
    for corpus_index, (corpus, corpus_result) in enumerate(source["results"].items()):
        records = []
        local_config = {**config, "seed": config["seed"] + corpus_index * 1000}
        for site in corpus_result["records"]:
            masks = [tuple(row["mask"]) for row in site["masks"]]
            row = {"query": site["query"], "row": site["row"]}
            for target in ("full_fidelity", "objective_efficiency"):
                y = np.asarray([record[target] for record in site["masks"]], dtype=np.float64)
                row[target] = {}
                for order in ("additive", "pairwise"):
                    row[target][order] = cross_validate(design(masks, order), y, local_config)
            records.append(row)
        results[corpus] = {"summary": aggregate(records), "records": records}

    finite = "full_fidelity"
    additive = all(results[c]["summary"][finite]["additive"]["median_r2"] >= .70
                   and results[c]["summary"][finite]["additive"]["median_regret"] <= .05 for c in results)
    pairwise = all(results[c]["summary"][finite]["pairwise"]["median_r2"] >= .80
                   and results[c]["summary"][finite]["pairwise"]["median_regret"] <= .03 for c in results)
    necessary = pairwise and all(
        results[c]["summary"][finite]["pairwise"]["median_r2"]
        - results[c]["summary"][finite]["additive"]["median_r2"] >= .20 for c in results
    )
    missing = any(results[c]["summary"][finite]["pairwise"]["median_r2"] < .50
                  or results[c]["summary"][finite]["pairwise"]["q90_regret"] > .10 for c in results)
    hypotheses = {
        "G0_additive_mask_grammar": {"supported": bool(additive)},
        "G1_pairwise_mask_grammar": {"supported": bool(pairwise)},
        "G2_pairwise_interactions_are_necessary": {"supported": bool(necessary)},
        "G3_higher_order_or_missing_variables": {"supported": bool(missing)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for corpus, color in (("fineweb", "#1769aa"), ("wikitext", "#b71c1c")):
        axes[0].plot([0, 1], [results[corpus]["summary"][finite][o]["median_r2"] for o in ("additive", "pairwise")], marker="o", color=color, label=corpus)
        axes[1].plot([0, 1], [results[corpus]["summary"][finite][o]["q90_regret"] for o in ("additive", "pairwise")], marker="o", color=color, label=corpus)
    for axis, title in zip(axes, ("Median held-out R²", "90th-percentile selection regret")):
        axis.set_xticks([0, 1], ["additive", "pairwise"]); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)

    lines = ["# Interaction grammar of the Layer-8 suffix masks", "",
             "Additive and pairwise set functions are fit separately at each site and evaluated on held-out three-layer masks. Selection regret asks whether the fitted surface identifies a good unseen mask, not just whether it predicts average scores.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Mask grammar](layer8_suffix_mask_grammar.png)", "",
              "| Corpus | Target | Model | Median R2 | 10th-percentile R2 | Median regret | 90th-percentile regret |",
              "|---|---|---|---:|---:|---:|---:|"]
    for corpus in results:
        for target in results[corpus]["summary"]:
            for order, row in results[corpus]["summary"][target].items():
                lines.append(f"| {corpus} | {target} | {order} | {row['median_r2']:.3f} | {row['q10_r2']:.3f} | {row['median_regret']:.3f} | {row['q90_regret']:.3f} |")
    lines += ["", "These are site-local descriptions. Pairwise success would identify a short local grammar but not a cross-context address rule. Pairwise failure rejects this polynomial set-function audit, not all compact tensor factorizations."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
