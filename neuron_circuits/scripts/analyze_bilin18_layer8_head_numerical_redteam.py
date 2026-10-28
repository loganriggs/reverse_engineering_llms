#!/usr/bin/env python3
"""Check whether head localization survives float32 Mobius cancellation."""

import json
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_head_numerical_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_head_numerical_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_HEAD_NUMERICAL_REDTEAM.md"


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    rows = source["records"]
    max_exact = max(row[side]["recomposition_error_to_exact_response"]
                    for row in rows for side in ("failure", "control"))
    same = all(row["failure"]["top_head"] == row["failure"]["recomposed_top_head"]
               for row in rows)
    share = float(np.median([row["failure"]["recomposed_top_absolute_share"] for row in rows]))
    support = float(np.median([row["failure"]["recomposed_effective_head_support"] for row in rows]))
    hypotheses = {
        "N0_error_is_cancellation_relative": {"supported": bool(max_exact <= 1e-4)},
        "N1_head_identity_is_numerically_stable": {"supported": bool(same)},
        "N2_localization_strength_is_stable": {"supported": bool(share >= .50 and support <= 3)},
        "N3_original_localization_was_numerical_artifact": {"supported": bool(not same or support > 5)},
    }
    summary = {"max_error_to_exact_response": max_exact,
               "all_failure_top_heads_unchanged": same,
               "median_recomposed_top_share": share,
               "median_recomposed_effective_support": support}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary}, indent=2))
    lines = ["# Numerical red team of third-order head localization", "",
             "The original exact-sum gate used error relative to a cancellation-small third-order residual. This audit scales error by the full exact response and independently defines projection shares against the recomposed sum of head terms.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum error relative to exact response: `{max_exact:.3e}`. All failure top heads unchanged: `{same}`. Median recomposed-reference top share: `{share:.3f}`. Median effective support: `{support:.2f}`.", "",
              "If localization survives while residual-relative identity misses, the discrepancy is float32 cancellation around a tiny residual rather than a different head decomposition."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
