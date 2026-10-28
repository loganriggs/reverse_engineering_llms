#!/usr/bin/env python3
"""Repeat leave-relation-out alignment across initializations of one task codebook."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from run_grammar_cross_seed_leave_relation_out import audit_pair


CONFIG = Path("configs/grammar_same_codebook_leave_relation_out.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_same_codebook_leave_relation_out.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_SAME_CODEBOOK_LEAVE_RELATION_OUT.md")
DIFFERENT = Path("artifacts/grammar_regimes/grammar_cross_seed_leave_relation_out.json")


def main():
    config = json.loads(CONFIG.read_text())
    loro_config = json.loads(Path(config["parent_config"]).read_text())
    base_config = json.loads(Path(loro_config["parent_config"]).read_text())
    assay_config = {**loro_config, "seed": config["seed"],
                    "alignment_r2_gate": config["alignment_r2_gate"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for source, target in permutations(config["runs"], 2):
        print("audit", source["name"], "->", target["name"], flush=True)
        results.append(audit_pair(source, target, assay_config, base_config, device))

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
    different = json.loads(DIFFERENT.read_text())["summary"]
    adjudicative = [layer for layer in (1, 2, 3) if valid[str(layer)]]
    early = [layer for layer in (1, 2) if valid[str(layer)]]
    s0 = bool(adjudicative)
    s1 = any(summary[str(layer)]["mapped_fidelity"] >= .50
             and summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["own_accuracy"] - .15 for layer in early)
    s2 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["wrong_fidelity"] + .20
             or summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["wrong_accuracy"] + .15 for layer in early)
    final = summary["3"]
    s3 = (valid["3"] and final["mapped_fidelity"] >= .70
          and final["target_answer_fidelity"] >= .70
          and final["target_answer_fidelity"] >= final["mapped_fidelity"] - .10)
    s4 = any(summary[str(layer)]["alignment_r2_mean"]
             >= different[str(layer)]["alignment_r2_mean"] + .50
             and summary[str(layer)]["mapped_fidelity"]
             >= different[str(layer)]["mapped_fidelity"] + .30
             for layer in adjudicative)
    hypotheses = {
        "S0_same_codebook_alignment_is_adjudicative": {"supported": s0, "valid_layers": adjudicative},
        "S1_rule_edit_is_initialization_invariant": {"supported": s1},
        "S2_same_rule_beats_wrong_rule": {"supported": s2},
        "S3_final_effect_matches_target_answer_interface": {"supported": s3},
        "S4_random_codebooks_break_linear_invariance": {"supported": s4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "valid_layers": valid, "summary": summary,
                                  "different_codebook_summary": different,
                                  "pairs": results}, indent=2))
    lines = ["# Same-codebook initialization invariance", "",
             "The grammar, data manifest, and random answer permutation are fixed; only initialization and training order differ. Alignment excludes the tested relation.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Alignment R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Target-answer fidelity | Different-codebook R2/fidelity | Adjudicative |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for layer in (1, 2, 3):
        row, old = summary[str(layer)], different[str(layer)]
        lines.append(
            f"| {layer} | {row['alignment_r2_mean']:.3f}/{row['alignment_r2_min']:.3f} | "
            f"{row['own_accuracy']:.3f}/{row['mapped_accuracy']:.3f}/{row['wrong_accuracy']:.3f} | "
            f"{row['mapped_fidelity']:.3f}/{row['wrong_fidelity']:.3f} | "
            f"{row['target_answer_fidelity']:.3f} | "
            f"{old['alignment_r2_mean']:.3f}/{old['mapped_fidelity']:.3f} | "
            f"{'yes' if valid[str(layer)] else 'no'} |"
        )
    lines += ["", "The comparison isolates task-codebook changes from initialization changes. Different training durations remain a nuisance variable, but all included models independently pass the same behavioral recall and template-transfer gate."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
