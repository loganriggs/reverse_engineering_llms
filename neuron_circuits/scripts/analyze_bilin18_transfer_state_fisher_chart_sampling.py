#!/usr/bin/env python3
"""Compare coordinate and behavioral stability of two Fisher-chart sketches."""

import json
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_transfer_state_fisher_chart_sampling_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart_sampling_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_FISHER_CHART_SAMPLING_REDTEAM.md"


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["base"]).read_text())
    high = json.loads(Path(config["highsample"]).read_text())
    rows = []
    for target in base["results"]:
        for boundary in base["results"][target]:
            first, second = base["results"][target][boundary], high["results"][target][boundary]
            rows.append({
                "target": target,
                "boundary": int(boundary),
                "base_overlap": first["sketch_overlap64_median"],
                "high_overlap": second["sketch_overlap64_median"],
                "base64": first["ranks"]["64"]["response_fidelity"],
                "high64": second["ranks"]["64"]["response_fidelity"],
                "base128": first["ranks"]["128"]["response_fidelity"],
                "high128": second["ranks"]["128"]["response_fidelity"],
                "static64": first["static64_fidelity"],
            })
    summary = {}
    for rank in (64, 128):
        first = np.asarray([row[f"base{rank}"] for row in rows])
        second = np.asarray([row[f"high{rank}"] for row in rows])
        summary[str(rank)] = {
            "spearman": spearman(first, second),
            "median_absolute_difference": float(np.median(np.abs(first - second))),
            "maximum_absolute_difference": float(np.max(np.abs(first - second))),
        }
    overlap_by_boundary = {}
    for boundary in sorted({row["boundary"] for row in rows}):
        selected = [row for row in rows if row["boundary"] == boundary]
        overlap_by_boundary[str(boundary)] = {
            "base": float(np.median([row["base_overlap"] for row in selected])),
            "highsample": float(np.median([row["high_overlap"] for row in selected])),
        }
    mean_overlap = float(np.mean([row["high_overlap"] for row in rows]))
    hypotheses = {
        "R1_rank64_behavior_is_sampling_stable": {
            "supported": summary["64"]["spearman"] >= .95 and summary["64"]["maximum_absolute_difference"] <= .05,
        },
        "R2_rank128_behavior_is_sampling_stable": {
            "supported": summary["128"]["spearman"] >= .90 and summary["128"]["maximum_absolute_difference"] <= .10,
        },
        "R3_more_samples_stabilize_literal_axes": {
            "supported": all(value["highsample"] >= value["base"] + .10 for value in overlap_by_boundary.values()),
        },
        "R4_behavior_is_more_stable_than_coordinates": {
            "supported": summary["64"]["spearman"] >= mean_overlap + .30,
        },
        "R5_highsample_chart_still_beats_static": {
            "supported": all(row["high64"] >= row["static64"] + .20 for row in rows),
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summary": summary,
        "mean_highsample_basis_overlap": mean_overlap,
        "overlap_by_boundary": overlap_by_boundary,
        "rows": rows,
    }, indent=2, allow_nan=False))
    lines = [
        "# Fisher-chart sampling red team", "",
        "The base 8-top/8-tail and independent 16-top/16-tail runs use different seeds and chart estimates. This post-run comparison separates stability of literal rank-64 axes from stability of their held-out causal performance.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Summary", "", "| Rank | Fidelity Spearman | Median / max absolute difference |", "|---:|---:|---:|"]
    for rank in (64, 128):
        row = summary[str(rank)]
        lines.append(f"| {rank} | {row['spearman']:.3f} | {row['median_absolute_difference']:.3f} / {row['maximum_absolute_difference']:.3f} |")
    lines += ["", f"Mean higher-sample literal rank-64 overlap: `{mean_overlap:.3f}`.", "", "| Boundary | Base overlap | Higher-sample overlap |", "|---:|---:|---:|"]
    for boundary, row in overlap_by_boundary.items():
        lines.append(f"| {boundary} | {row['base']:.3f} | {row['highsample']:.3f} |")
    lines += [
        "", "## Guardrails", "",
        "Stable scalar fidelity does not mean the two charts predict identical logit-effect directions candidate by candidate; only aggregate fidelity cells are compared here. Conversely, low canonical overlap can be caused by broad or near-degenerate eigenspectra, so it does not by itself invalidate the behaviorally defined subspace family. Rank-128 maximum-difference failure would keep fine-grained high-rank claims provisional.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "summary": summary, "mean_overlap": mean_overlap}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
