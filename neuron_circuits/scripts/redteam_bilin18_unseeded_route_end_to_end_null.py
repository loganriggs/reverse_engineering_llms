#!/usr/bin/env python3
"""Repeat the finite suffix audit with 32 destroyed-binding nulls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bilin18_unseeded_route_end_to_end_causality as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_unseeded_route_end_to_end_null_redteam.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_end_to_end_null_redteam.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_END_TO_END_NULL_REDTEAM.md"


if __name__ == "__main__":
    experiment.main()
