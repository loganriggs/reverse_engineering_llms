import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_stratified_natural_mlp_census.py"
spec = importlib.util.spec_from_file_location("stratified_natural_mlp_census", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_scalar_fidelity_exact_for_all_channels():
    values = np.asarray([[1.0, 2.0], [-1.0, 3.0]])
    assert np.isclose(module.scalar_fidelity(values, np.ones(2, dtype=bool), np.arange(2)), 1.0)


def test_jaccard():
    assert np.isclose(module.jaccard([1, 2], [2, 3]), 1 / 3)


def test_oracle_k90_counts_absolute_mass():
    values = np.asarray([[9.0, 1.0, 0.0], [1.0, 1.0, 8.0]])
    assert module.oracle_k90(values).tolist() == [1, 2]
