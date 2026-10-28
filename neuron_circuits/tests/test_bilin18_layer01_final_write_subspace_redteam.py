from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_write_subspace_redteam import bootstrap  # noqa: E402


def test_bootstrap_constant():
    assert bootstrap([2,2,2],100,np.random.default_rng(1))==[2.,2.,2.]
