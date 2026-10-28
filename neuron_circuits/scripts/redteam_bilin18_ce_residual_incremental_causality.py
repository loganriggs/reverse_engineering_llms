#!/usr/bin/env python3
"""Repeat incremental causal coverage against 32 random complement bases."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bilin18_ce_residual_incremental_causality as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_ce_residual_incremental_causality_redteam.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_ce_residual_incremental_causality_redteam.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_CE_RESIDUAL_INCREMENTAL_CAUSALITY_REDTEAM.md"


if __name__ == "__main__":
    experiment.main()
