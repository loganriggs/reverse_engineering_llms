#!/usr/bin/env python3
"""High-count replication of the operation-family tournament."""

from pathlib import Path
import run_grammar_operation_family_tournament as tournament

tournament.CONFIG = Path("configs/grammar_operation_family_effectful_replication.json")
tournament.OUTPUT = Path("artifacts/grammar_regimes/grammar_operation_family_effectful_replication.json")
tournament.REPORT = Path("artifacts/grammar_regimes/GRAMMAR_OPERATION_FAMILY_EFFECTFUL_REPLICATION.md")
tournament.main()
