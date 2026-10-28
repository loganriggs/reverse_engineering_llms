import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_stratified_chart_factor_symmetry.py"
spec = importlib.util.spec_from_file_location("chart_factor_symmetry", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_orthogonal_grid_is_orthogonal_and_complete():
    grid = module.orthogonal_grid(8)
    assert len(grid) == 16
    for _, matrix in grid:
        assert np.allclose(matrix @ matrix.T, np.eye(2), atol=1e-8)
