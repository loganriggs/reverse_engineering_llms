#!/usr/bin/env python3
"""Run the prospectively locked 128-site replication of the logit/CE program."""

from pathlib import Path

import run_bilin18_layer16_logit_ce_program as experiment
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


experiment.CONFIG = Path("configs/bilin18_layer16_logit_ce_highcount.json")
experiment.OUTPUT = TENSOR_ROOT / "bilin18_layer16_logit_ce_highcount.json"
experiment.REPORT = TENSOR_ROOT / "BILIN18_LAYER16_LOGIT_CE_HIGHCOUNT.md"


if __name__ == "__main__":
    experiment.main()
