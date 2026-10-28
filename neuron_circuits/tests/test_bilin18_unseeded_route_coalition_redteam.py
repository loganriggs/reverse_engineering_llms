import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_unseeded_route_coalitions.py"
spec = importlib.util.spec_from_file_location("unseeded_route_coalition_redteam", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_center_separation():
    centers = np.eye(3)
    assert module.maximum_off_diagonal_cosine(centers) == 0


def test_parameter_matched_subspace_exact_on_plane():
    discovery = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=float)
    replication = np.asarray([[1, 1, 0]], dtype=float)
    assert abs(module.subspace_replication_fidelity(discovery, replication, 2) - 1) < 1e-12
