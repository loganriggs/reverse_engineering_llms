#!/usr/bin/env python3
"""High-count replication of the pure/mixed finite gate-depth comparison."""

from pathlib import Path

import run_grammar_pure_mixed_gate_depth_comparison as experiment


experiment.CONFIG=Path("configs/grammar_pure_mixed_gate_depth_effectful_replication.json")
experiment.OUTPUT=Path("artifacts/grammar_regimes/grammar_pure_mixed_gate_depth_effectful_replication.json")
experiment.REPORT=Path("artifacts/grammar_regimes/GRAMMAR_PURE_MIXED_GATE_DEPTH_EFFECTFUL_REPLICATION.md")


if __name__=="__main__":
    experiment.main()
