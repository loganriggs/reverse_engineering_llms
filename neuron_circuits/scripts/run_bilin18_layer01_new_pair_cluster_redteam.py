#!/usr/bin/env python3
"""Pair-clustered bootstrap and family-label null for new-pair Layer-0/1 banks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_new_pair_cluster_redteam.json")
OUTPUT = TENSOR_ROOT / "layer01_new_pair_cluster_redteam.json"
REPORT = TENSOR_ROOT / "LAYER01_NEW_PAIR_CLUSTER_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer01_new_pair_cluster_redteam.png"


def rankdata(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:stop]] = (start + stop - 1) / 2
    return ranks


def spearman(first: list[float], second: list[float]) -> float:
    left, right = rankdata(first), rankdata(second)
    left -= left.mean(); right -= right.mean()
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-30))


def aggregate_pairs(records: list[dict], families: list[str]) -> list[dict]:
    rows = []
    for pair in sorted({record["pair"] for record in records}):
        selected = [record for record in records if record["pair"] == pair]
        family = selected[0]["family"]
        effects = {
            name: float(np.median([record["family_effects"][name] for record in selected]))
            for name in families
        }
        correct = effects[family]
        wrong = float(np.median([value for name, value in effects.items() if name != family]))
        rows.append({
            "pair": pair,
            "family": family,
            "effects": effects,
            "correct": correct,
            "wrong_median": wrong,
            "advantage": correct - wrong,
            "top1": max(effects, key=effects.get),
            "strict": float(np.median([record["strict"] for record in selected])),
        })
    return rows


def mapping_stat(rows: list[dict], labels: list[str], families: list[str]) -> float:
    advantages = []
    for row, label in zip(rows, labels):
        correct = row["effects"][label]
        wrong = np.median([value for name, value in row["effects"].items() if name != label])
        advantages.append(correct - wrong)
    return float(np.median(advantages))


def summarize(rows: list[dict], families: list[str], rng: np.random.Generator, config: dict) -> dict:
    advantages = np.asarray([row["advantage"] for row in rows])
    boot = []
    for _ in range(config["bootstrap_replicates"]):
        sample = rng.integers(0, len(rows), len(rows))
        boot.append(float(np.median(advantages[sample])))
    labels = [row["family"] for row in rows]
    null = []
    for _ in range(config["permutation_replicates"]):
        null.append(mapping_stat(rows, list(rng.permutation(labels)), families))
    observed = float(np.median(advantages))
    return {
        "median_advantage": observed,
        "pair_win_rate": float(np.mean(advantages > 0)),
        "top1_accuracy": float(np.mean([row["top1"] == row["family"] for row in rows])),
        "strict_positive_rate": float(np.mean([row["strict"] > 0 for row in rows])),
        "bootstrap_low": float(np.quantile(boot, .025)),
        "bootstrap_high": float(np.quantile(boot, .975)),
        "null_median": float(np.median(null)),
        "null_p99": float(np.quantile(null, .99)),
        "empirical_p": float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))),
        "families": {
            family: {
                "median_advantage": float(np.median([row["advantage"] for row in rows if row["family"] == family])),
                "win_rate": float(np.mean([row["advantage"] > 0 for row in rows if row["family"] == family])),
                "top1_accuracy": float(np.mean([row["top1"] == family for row in rows if row["family"] == family])),
            }
            for family in families
        },
    }


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_output"]).read_text())
    families = list(source["layers"]["0"]["summaries"]["fineweb"]["families"])
    rng = np.random.default_rng(config["seed"])
    results = {}
    pair_rows = {}
    for layer in ("0", "1"):
        results[layer], pair_rows[layer] = {}, {}
        for corpus in ("fineweb", "tinystories"):
            records = [row for row in source["layers"][layer]["records"] if row["corpus"] == corpus]
            rows = aggregate_pairs(records, families)
            pair_rows[layer][corpus] = rows
            results[layer][corpus] = summarize(rows, families, rng, config)
        fine = {row["pair"]: row["advantage"] for row in pair_rows[layer]["fineweb"]}
        tiny = {row["pair"]: row["advantage"] for row in pair_rows[layer]["tinystories"]}
        common = sorted(set(fine) & set(tiny))
        results[layer]["cross_corpus_spearman"] = spearman(
            [fine[pair] for pair in common], [tiny[pair] for pair in common],
        )

    hypotheses = {
        "R0_pair_clustered_advantage_is_material": {"supported": all(results[l][c]["median_advantage"] >= .03 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R1_most_pairs_prefer_correct_family": {"supported": all(results[l][c]["pair_win_rate"] >= .65 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R2_correct_bank_is_often_top1": {"supported": all(results[l][c]["top1_accuracy"] >= .5 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R3_family_mapping_beats_permutation": {"supported": all(results[l][c]["empirical_p"] <= .01 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R4_bootstrap_interval_excludes_zero": {"supported": all(results[l][c]["bootstrap_low"] > 0 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R5_pair_difficulty_is_cross_corpus_stable": {"supported": all(results[l]["cross_corpus_spearman"] >= .3 for l in ("0","1"))},
        "R6_result_is_not_one_family": {"supported": all(sum(results[l][c]["families"][f]["median_advantage"] > 0 for f in families) >= 4 for l in ("0","1") for c in ("fineweb","tinystories"))},
        "R7_strict_core_is_pairwise_positive": {"supported": all(results[l][c]["strict_positive_rate"] >= .65 for l in ("0","1") for c in ("fineweb","tinystories"))},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"results":results,"pairs":pair_rows},indent=2))
    lines=["# Pair-clustered red-team of prospective Layer-0/1 semantic banks","","The three contexts for each of seventeen new lexical substitutions are aggregated before inference. Pair-bootstrap intervals, size-preserving family-label permutations, top-1 bank classification, per-family replication, and cross-corpus pair consistency test whether the example-level result was inflated by correlated contexts.","","## Locked hypotheses",""]
    for name,item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["","![Pair-clustered red-team](layer01_new_pair_cluster_redteam.png)","","| Layer | Corpus | Median advantage (95% pair bootstrap) | Pair win | Top-1 family | Null p99 / p | Strict positive |","|---:|---|---:|---:|---:|---:|---:|"]
    for layer in ("0","1"):
        for corpus in ("fineweb","tinystories"):
            row=results[layer][corpus]
            lines.append(f"| {layer} | {corpus} | {row['median_advantage']:.3f} ({row['bootstrap_low']:.3f}, {row['bootstrap_high']:.3f}) | {row['pair_win_rate']:.3f} | {row['top1_accuracy']:.3f} | {row['null_p99']:.3f} / {row['empirical_p']:.4f} | {row['strict_positive_rate']:.3f} |")
    lines += ["",f"Cross-corpus pair-advantage Spearman: Layer 0 **{results['0']['cross_corpus_spearman']:.3f}**; Layer 1 **{results['1']['cross_corpus_spearman']:.3f}**.","","Permutation significance tests the declared family mapping, not whether the family names are the only possible abstraction. A generic bank can remain competitive even when the correct named bank reliably beats wrong named banks."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for axis,layer in zip(axes,("0","1")):
        fine={r['pair']:r['advantage'] for r in pair_rows[layer]['fineweb']}; tiny={r['pair']:r['advantage'] for r in pair_rows[layer]['tinystories']}; pairs=sorted(fine)
        axis.scatter([fine[p] for p in pairs],[tiny[p] for p in pairs]); axis.axhline(0,color='black',linewidth=.7); axis.axvline(0,color='black',linewidth=.7); axis.set_xlabel('FineWeb pair advantage'); axis.set_ylabel('TinyStories pair advantage'); axis.set_title(f"Layer {layer}, rho={results[layer]['cross_corpus_spearman']:.2f}"); axis.grid(alpha=.2)
    fig.savefig(FIGURE,dpi=180); plt.close(fig)
    print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__ == "__main__": main()
