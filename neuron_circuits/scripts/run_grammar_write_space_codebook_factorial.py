#!/usr/bin/env python3
"""Run the leave-relation-out MLP-write assay across different answer codebooks."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from run_grammar_write_space_invariance import audit_pair


CONFIG = Path("configs/grammar_write_space_codebook_factorial.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_write_space_codebook_factorial.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_WRITE_SPACE_CODEBOOK_FACTORIAL.md")
SAME = Path("artifacts/grammar_regimes/grammar_write_space_invariance.json")


def main():
    config = json.loads(CONFIG.read_text())
    write_config = json.loads(Path(config["parent_config"]).read_text())
    same_config = json.loads(Path(write_config["parent_config"]).read_text())
    loro = json.loads(Path(same_config["parent_config"]).read_text())
    base = json.loads(Path(loro["parent_config"]).read_text())
    original = json.loads(Path(base["parent_config"]).read_text())
    runs = [run for run in original["runs"] if run["rho"] == 0.0]
    assay_config = {**write_config, "seed": config["seed"],
                    "alignment_r2_gate": config["alignment_r2_gate"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for source, target in permutations(runs, 2):
        print("audit", source["name"], "->", target["name"], flush=True)
        results.append(audit_pair(source, target, assay_config, loro, base, device))
    summary, valid = {}, {}
    for layer in (1, 2, 3):
        alignment_r2 = [pair["alignment"][relation][layer]["r2"]
                        for pair in results for relation in range(4)]
        valid[str(layer)] = min(alignment_r2) >= config["alignment_r2_gate"]
        values = [pair["layers"][str(layer)] for pair in results]
        keys = [key for key in values[0] if key != "transitions"]
        summary[str(layer)] = {key: float(np.mean([row[key] for row in values])) for key in keys}
        summary[str(layer)]["alignment_r2_mean"] = float(np.mean(alignment_r2))
        summary[str(layer)]["alignment_r2_min"] = float(np.min(alignment_r2))
    same = json.loads(SAME.read_text())["summary"]
    c0 = not any(valid.values())
    c1 = (same["3"]["alignment_r2_mean"] >= summary["3"]["alignment_r2_mean"] + .30
          and same["3"]["mapped_fidelity"] >= summary["3"]["mapped_fidelity"] + .30)
    c2 = summary["3"]["target_answer_fidelity"] >= .70
    c3 = summary["3"]["mapped_accuracy"] < summary["3"]["wrong_accuracy"] + .15
    hypotheses = {
        "C0_different_codebooks_break_unseen_relation_write_alignment": {"supported": c0},
        "C1_late_invariance_is_codebook_relative": {"supported": c1},
        "C2_target_internal_answer_interface_survives": {"supported": c2},
        "C3_wrong_rule_control_remains_noncausal": {"supported": c3},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "valid_layers": valid, "summary": summary,
                                  "same_codebook_summary": same, "pairs": results}, indent=2))
    lines = ["# MLP-write codebook factorial", "",
             "The symmetry-aware leave-one-relation-out assay is repeated across models with independently randomized answer permutations.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Different-codebook R2 mean/min | Same-codebook R2 mean/min | Different/same mapped fidelity | Mapped/wrong accuracy | Target-answer fidelity | Adjudicative |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for layer in (1, 2, 3):
        row, matched = summary[str(layer)], same[str(layer)]
        lines.append(
            f"| {layer} | {row['alignment_r2_mean']:.3f}/{row['alignment_r2_min']:.3f} | "
            f"{matched['alignment_r2_mean']:.3f}/{matched['alignment_r2_min']:.3f} | "
            f"{row['mapped_fidelity']:.3f}/{matched['mapped_fidelity']:.3f} | "
            f"{row['mapped_accuracy']:.3f}/{row['wrong_accuracy']:.3f} | "
            f"{row['target_answer_fidelity']:.3f} | {'yes' if valid[str(layer)] else 'no'} |"
        )
    lines += ["", "The random output convention is part of the implemented task. A codebook-relative result means the final interface is reusable within that convention, not that it lacks structure."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
