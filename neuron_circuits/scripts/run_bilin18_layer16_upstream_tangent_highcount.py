#!/usr/bin/env python3
"""Run the locked 1,024-site selection replication for natural upstream tangents."""

from pathlib import Path

import run_bilin18_layer16_upstream_tangent_channels as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_layer16_upstream_tangent_highcount.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_layer16_upstream_tangent_highcount.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_LAYER16_UPSTREAM_TANGENT_HIGHCOUNT.md"


if __name__ == "__main__": experiment.main()
