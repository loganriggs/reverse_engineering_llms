#!/usr/bin/env python3
"""Decode the untouched replication panel using frozen discovery pole axes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bilin18_ce_residual_pole_semantic_discovery as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_ce_residual_pole_semantic_replication.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_ce_residual_pole_semantic_replication.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_CE_RESIDUAL_POLE_SEMANTIC_REPLICATION.md"
experiment.PANEL = "replication"


if __name__ == "__main__":
    experiment.main()
