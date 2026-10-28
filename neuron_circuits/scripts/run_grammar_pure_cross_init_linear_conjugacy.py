#!/usr/bin/env python3
"""Run pure-rule residual alignment with the shared conjugacy analyzer."""

from pathlib import Path
import run_grammar_cross_init_linear_conjugacy as analysis

analysis.CONFIG = Path("configs/grammar_pure_cross_init_linear_conjugacy.json")
analysis.OUTPUT = Path("artifacts/grammar_regimes/grammar_pure_cross_init_linear_conjugacy.json")
analysis.REPORT = Path("artifacts/grammar_regimes/GRAMMAR_PURE_CROSS_INIT_LINEAR_CONJUGACY.md")
analysis.main()
