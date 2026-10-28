import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bilin18_known_ontology_residual_finite.py"
spec = importlib.util.spec_from_file_location("known_ontology_residual_finite", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_fixed_ridge_predicts_linear_map():
    rng = np.random.default_rng(4); x = rng.normal(size=(100, 4)); y = x @ rng.normal(size=(4, 7)) + 2
    model = module.fit_fixed_ridge(x, y, 1e-8); prediction = module.ridge_predict(model, x)
    assert np.square(y - prediction).mean() < 1e-12


def test_lookup_uses_row_position_pair():
    metadata = {"global_row": np.asarray([3, 3, 4]), "position": np.asarray([7, 9, 7])}
    assert module.make_lookup(metadata) == {(3, 7): 0, (3, 9): 1, (4, 7): 2}
