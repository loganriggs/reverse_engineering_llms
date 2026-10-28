#!/usr/bin/env python3
"""Run the internal-controller audit without a template-coordinate shift."""

from pathlib import Path

import run_grammar_mixed_chart_controller as experiment


experiment.CONFIG = Path("configs/grammar_mixed_chart_controller_same_template.json")
experiment.OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_chart_controller_same_template.json")
experiment.REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_CONTROLLER_SAME_TEMPLATE.md")


if __name__ == "__main__":
    experiment.main()
