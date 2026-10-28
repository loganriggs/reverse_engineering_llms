#!/usr/bin/env python3
"""Compare joint and position-collapsed charts at runtime coordinate budgets."""

import json
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_transfer_state_joint_budget_analysis.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_joint_budget_analysis.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_JOINT_BUDGET_ANALYSIS.md"


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    rows = []
    for target, target_rows in source["results"].items():
        for boundary, row in target_rows.items():
            rows.append({
                "target": target,
                "boundary": int(boundary),
                "collapsed": {rank: value["response_fidelity"] for rank, value in row["methods"]["collapsed"].items()},
                "joint": {rank: value["response_fidelity"] for rank, value in row["methods"]["joint"].items()},
            })
    length = config["sequence_length"]
    b1 = all(row["joint"]["64"] >= row["collapsed"]["1"] + .30 for row in rows)
    b2 = all(row["joint"]["128"] >= row["collapsed"]["1"] + .40 for row in rows)
    b3 = all(row["joint"]["64"] >= row["collapsed"]["4"] for row in rows)
    b4 = all(row["joint"]["64"] >= .75 for row in rows if row["boundary"] == 6)
    budgets = [1, 4, 16, 64, 128]
    frontier = {}
    b5 = True
    for budget in budgets:
        joint_values = [row["joint"][str(rank)] for row in rows for rank in budgets if rank <= budget]
        collapsed_ranks = [rank for rank in budgets if rank * length <= budget]
        best_joint = float(np.mean([
            max(row["joint"][str(rank)] for rank in budgets if rank <= budget)
            for row in rows
        ]))
        best_collapsed = None
        if collapsed_ranks:
            best_collapsed = float(np.mean([
                max(row["collapsed"][str(rank)] for rank in collapsed_ranks)
                for row in rows
            ]))
            b5 = b5 and best_joint > best_collapsed
        frontier[str(budget)] = {
            "joint_mean_fidelity": best_joint,
            "collapsed_mean_fidelity": best_collapsed,
            "collapsed_available_ranks": collapsed_ranks,
        }
    hypotheses = {
        "B1_joint64_beats_more_expensive_collapsed1": {"supported": b1},
        "B2_joint128_beats_coordinate_matched_collapsed1": {"supported": b2},
        "B3_joint64_beats_collapsed4_despite_eighth_coordinates": {"supported": b3},
        "B4_joint64_is_useful_at_early_boundary": {"supported": b4},
        "B5_joint_frontier_is_coordinate_pareto_superior": {"supported": b5},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "frontier": frontier,
        "rows": rows,
    }, indent=2, allow_nan=False))
    lines = [
        "# Runtime-coordinate budget for joint and collapsed charts", "",
        "A collapsed rank-r residual chart supplies r coefficients independently at each of 128 token positions, for 128r runtime coordinates. A joint rank-r sequence-field chart supplies only r coordinates total. This post-same-rank analysis compares causal fidelity under that execution budget. Chart-generation and declaration costs are discussed separately.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Coordinate frontier", "", "| Total coordinates | Best joint mean fidelity | Best collapsed mean fidelity | Collapsed ranks available |", "|---:|---:|---:|---|"]
    for budget, row in frontier.items():
        collapsed = "—" if row["collapsed_mean_fidelity"] is None else f"{row['collapsed_mean_fidelity']:.3f}"
        lines.append(f"| {budget} | {row['joint_mean_fidelity']:.3f} | {collapsed} | {row['collapsed_available_ranks']} |")
    lines += ["", "| Target | Boundary | Collapsed r1 (128 coords) | Collapsed r4 (512) | Joint r64 (64) | Joint r128 (128) |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['target']} | {row['boundary']} | {row['collapsed']['1']:.3f} | {row['collapsed']['4']:.3f} | {row['joint']['64']:.3f} | {row['joint']['128']:.3f} |")
    lines += [
        "", "## Guardrails", "",
        "Runtime coordinate count is not total MDL. A literal joint basis contains sequence_length × residual_dimension × rank numbers, while a collapsed basis is much cheaper to store. The joint basis is useful only because it is generated from the tensor-network Fisher program rather than transmitted literally. Compute cost, generator declaration, and residual error must all be included before calling either representation shorter overall.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "frontier": frontier}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
