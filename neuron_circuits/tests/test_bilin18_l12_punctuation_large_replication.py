import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replicate_bilin18_l12_punctuation_large_panel.py"
spec = importlib.util.spec_from_file_location("l12_punctuation_large", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_fidelity_exact():
    values = np.asarray([1.0, -2.0])
    assert np.isclose(module.fidelity(values, values), 1.0)


def test_cluster_bootstrap_returns_requested_draws():
    values = module.cluster_bootstrap(np.asarray([1, 1, 2]), np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]), 10, np.random.default_rng(1))
    assert len(values) == 10
    assert np.allclose(values, 1.0)
