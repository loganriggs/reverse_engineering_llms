#!/usr/bin/env python3
"""Run the preregistered high-count version of the 546M MLP-path assay."""

from pathlib import Path

import run_bilin18_mlp_path_symmetry_replication as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_mlp_path_highcount_replication.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_mlp_path_highcount_replication.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_MLP_PATH_HIGHCOUNT_REPLICATION.md"


if __name__ == "__main__":
    experiment.main()
