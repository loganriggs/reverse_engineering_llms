import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_bus_aware_function_templates import rank_summary


def test_rank_summary_recovers_known_spectrum():
    summary = rank_summary(np.diag([9.0, 1.0, 0.0]))
    assert summary["rank50"] == 1
    assert summary["rank90"] == 1
    assert abs(summary["top16_energy"] - 1.0) < 1e-9
