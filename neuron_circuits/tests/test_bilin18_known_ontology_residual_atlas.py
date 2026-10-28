import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_known_ontology_residual_atlas.py"
spec = importlib.util.spec_from_file_location("known_ontology_residual", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_ridge_fit_generalizes_exact_linear_map():
    rng = np.random.default_rng(2); x = rng.normal(size=(200, 4)); weight = rng.normal(size=(4, 6)); y = x @ weight + 3
    train_prediction, test_prediction, _, _ = module.ridge_fit(x[:150], y[:150], x[150:], [1e-8, 1e-4], np.arange(150))
    assert module.multivariate_r2(y[:150], train_prediction, y[:150].mean(0)) > .999
    assert module.multivariate_r2(y[150:], test_prediction, y[:150].mean(0)) > .999


def test_token_lookup_uses_smoothed_means():
    ids = np.asarray([0, 0, 1, 1]); values = np.asarray([[1.0], [3.0], [5.0], [7.0]])
    prediction, coverage = module.token_lookup(ids, values, np.asarray([0, 1, 2]), 0.0)
    assert np.allclose(prediction[:2, 0], [2, 6]) and coverage == 2 / 3


def test_axis_matching_handles_sign_and_permutation():
    first = np.eye(3); second = first[[2, 0, 1]] * np.asarray([-1, 1, -1])[:, None]
    permutation, values = module.match_axes(first, second)
    assert permutation.tolist() == [1, 2, 0]
    assert np.allclose(np.abs(values), 1)
