import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw  # noqa: E402


def test_powerlaw_fit_and_top_excess():
    ranks = np.arange(1, 800, dtype=np.float64)
    values = 5 * ranks ** -0.7
    values[0] *= 3
    result = fit_powerlaw(values, 33, 512, 4)
    assert abs(result["alpha"] - 0.7) < 1e-10
    assert result["r_squared"] > 0.999999
    assert result["top_log_excess"][0] > 1.09
    assert abs(result["top_log_excess"][1]) < 1e-10
