#!/usr/bin/env python3
"""Scattered-row red team for dominant destination-Fisher semantics."""

from pathlib import Path

import run_bilin18_destination_fisher_mode_semantics as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG=Path("configs/bilin18_destination_fisher_mode_semantics_scattered.json")
experiment.OUTPUT=TENSOR_ROOT/"bilin18_destination_fisher_mode_semantics_scattered.json"
experiment.REPORT=TENSOR_ROOT/"BILIN18_DESTINATION_FISHER_MODE_SEMANTICS_SCATTERED.md"


if __name__=="__main__":experiment.main()
