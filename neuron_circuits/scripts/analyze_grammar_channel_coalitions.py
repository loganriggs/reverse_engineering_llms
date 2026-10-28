#!/usr/bin/env python3
"""Analyze semantic and global reuse of per-site native channel coalitions."""

import json
from pathlib import Path

import numpy as np


CONFIG = Path("configs/grammar_channel_coalition_structure.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_channel_coalition_structure.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_CHANNEL_COALITION_STRUCTURE.md")


def analyze(row):
    selections = np.asarray(row["site_top32_channels"]); sites, count = selections.shape
    indicator = np.zeros((sites, 192), dtype=bool)
    indicator[np.arange(sites)[:, None], selections] = True
    overlap = indicator.astype(np.int32) @ indicator.astype(np.int32).T / count
    ii, jj = np.triu_indices(sites, 1)
    relations = np.asarray(row["evaluation_relations"]); answers = np.asarray(row["evaluation_rule_answers"])
    frequency = indicator.sum(0).astype(float); probability = frequency / frequency.sum()
    positive = probability > 0
    effective = float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))
    fixed = np.asarray(row["top32_channels"])
    fixed_capture = float(indicator[:, fixed].sum(1).mean() / count)
    def grouped(mask): return float(overlap[ii[mask], jj[mask]].mean())
    same_relation = relations[ii] == relations[jj]; same_answer = answers[ii] == answers[jj]
    return {"mean_overlap": float(overlap[ii, jj].mean()), "effective_selected_channels": effective,
            "fixed_bank_capture": fixed_capture,
            "same_relation_overlap": grouped(same_relation), "different_relation_overlap": grouped(~same_relation),
            "relation_advantage": grouped(same_relation) - grouped(~same_relation),
            "same_answer_overlap": grouped(same_answer), "different_answer_overlap": grouped(~same_answer),
            "answer_advantage": grouped(same_answer) - grouped(~same_answer)}


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["input"]).read_text())
    runs = [dict(seed=row["seed"], rho=row["rho"], **analyze(row)) for row in source["runs"]]
    def get(seed, rho): return next(row for row in runs if row["seed"] == seed and row["rho"] == rho)
    s0 = all(get(seed, .5)["mean_overlap"] <= .8 * get(seed, 0.0)["mean_overlap"] for seed in (0, 1))
    s1 = all(get(seed, .5)["effective_selected_channels"] >= 1.2 * get(seed, 0.0)["effective_selected_channels"] for seed in (0, 1))
    s2 = all(get(seed, 0.0)["relation_advantage"] >= .05 for seed in (0, 1))
    s3 = all(get(seed, .5)["relation_advantage"] <= .03 for seed in (0, 1))
    s4 = all(get(seed, 0.0)["fixed_bank_capture"] >= .75 for seed in (0, 1))
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (s0, s1, s2, s3, s4))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Structure of per-site bilinear channel coalitions", "",
             "Each site contributes its 32 largest exact final-block channel effects. Overlap is intersection divided by 32. The entropy-effective bank size measures how broadly selection frequency is distributed across all 192 channels. Relation and answer comparisons use every held-out pair of sites.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | rho | Mean overlap | Effective bank | Fixed-bank capture | Relation advantage | Answer advantage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        lines.append(f"| {row['seed']} | {row['rho']} | {row['mean_overlap']:.3f} | {row['effective_selected_channels']:.1f} | "
                     f"{row['fixed_bank_capture']:.3f} | {row['relation_advantage']:.3f} | {row['answer_advantage']:.3f} |")
    lines += ["", "Selection is based on effect norm and is not an optimal subset under cancellation. Pairwise sites within one trained model are correlated; the two seeds, not site pairs, are the replication units."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
