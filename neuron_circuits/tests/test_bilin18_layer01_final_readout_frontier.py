from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_readout_frontier import first_threshold  # noqa: E402


def test_first_threshold_uses_declared_frontier():
    assert first_threshold([.1,.6,.95],[32,64,128],.5)==64
    assert first_threshold([.1,.2],[32,64],.9) is None
