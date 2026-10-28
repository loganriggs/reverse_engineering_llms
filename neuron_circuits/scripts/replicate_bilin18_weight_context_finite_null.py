#!/usr/bin/env python3
"""Replicate weight/context finite specificity on a second fresh FineWeb panel."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redteam_bilin18_weight_context_finite_null as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_weight_context_finite_null_replication.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_weight_context_finite_null_replication.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_CONTEXT_FINITE_NULL_REPLICATION.md"


if __name__ == "__main__":
    experiment.main()
