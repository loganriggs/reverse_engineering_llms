#!/usr/bin/env python3
"""Run the locked pure-rule cross-init comparison with the shared analyzer."""

from pathlib import Path
import run_grammar_mixed_gate_cross_init as analysis

analysis.CONFIG = Path("configs/grammar_pure_gate_cross_init.json")
analysis.OUTPUT = Path("artifacts/grammar_regimes/grammar_pure_gate_cross_init.json")
analysis.REPORT = Path("artifacts/grammar_regimes/GRAMMAR_PURE_GATE_CROSS_INIT.md")
analysis.main()
