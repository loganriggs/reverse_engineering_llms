import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redteam_bilin18_l12_punctuation_lookup.py"
spec = importlib.util.spec_from_file_location("l12_punctuation_lookup", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_token_lookup_recovers_seen_means():
    prediction = module.token_lookup(np.asarray([1, 1, 2]), np.asarray([1.0, 3.0, 5.0]), np.asarray([1, 2, 3]))
    assert np.allclose(prediction, [2.0, 5.0, 3.0])


def test_unique_row_selection():
    score = np.asarray([4.0, 3.0, 2.0, 1.0]); eligible = np.arange(4); rows = np.asarray([1, 1, 2, 3])
    assert module.select(score, eligible, 3, True, rows).tolist() == [0, 2, 3]
