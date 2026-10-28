import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_contextual_source_variables import make_schemes


def test_context_schemes_are_causal_and_shape_preserving():
    base = np.arange(32) % 21
    ids = np.asarray([[1, 2, 1, 3, 4, 1, 2, 5, 1]])
    schemes = make_schemes(base, 7)
    for _, (function, count) in schemes.items():
        categories = function(ids)
        assert categories.shape == ids.shape
        assert categories.min() >= 0
        assert categories.max() < count
    # Changing the final token cannot alter any earlier source category.
    changed = ids.copy(); changed[0, -1] = 6
    for _, (function, _) in schemes.items():
        assert np.array_equal(function(ids)[:, :-1], function(changed)[:, :-1])
