#!/usr/bin/env python3
"""Run the destination-Fisher experiment for unrelated target L4H1."""

from pathlib import Path

import run_bilin18_destination_fisher_geometry as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG=Path("configs/bilin18_destination_fisher_geometry_third_target.json")
experiment.OUTPUT=TENSOR_ROOT/"bilin18_destination_fisher_geometry_third_target.json"
experiment.TENSORS=TENSOR_ROOT/"bilin18_destination_fisher_geometry_third_target.pt"
experiment.REPORT=TENSOR_ROOT/"BILIN18_DESTINATION_FISHER_GEOMETRY_THIRD_TARGET.md"


if __name__=="__main__":experiment.main()
