#!/usr/bin/env python3
"""Run the destination-Fisher experiment with L2H5 as the target."""

from pathlib import Path

import run_bilin18_destination_fisher_geometry as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_destination_fisher_geometry_reverse.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_destination_fisher_geometry_reverse.json"
experiment.TENSORS = TENSOR_ROOT / "bilin18_destination_fisher_geometry_reverse.pt"
experiment.REPORT = TENSOR_ROOT / "BILIN18_DESTINATION_FISHER_GEOMETRY_REVERSE.md"


if __name__ == "__main__":
    experiment.main()
