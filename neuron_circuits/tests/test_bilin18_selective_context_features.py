import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_contextual_source_variables import make_schemes
from run_bilin18_selective_context_features import selective_scheme


def test_selective_scheme_only_splits_named_base_categories():
    base = np.arange(64) % 21
    crossed = make_schemes(base, 3)
    function, count = selective_scheme(base, crossed, [(1, "recent_repeat", 2)])
    ids = np.asarray([[1, 2, 1, 3, 3]])
    result = function(ids)
    assert count == 22
    assert np.array_equal(result[0, [1, 3, 4]], base[ids][0, [1, 3, 4]])
    assert result[0, 2] == 21
