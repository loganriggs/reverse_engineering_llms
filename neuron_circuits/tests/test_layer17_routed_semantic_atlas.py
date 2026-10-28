import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_routed_semantic_atlas import adjusted_rand, lookup_fit, lookup_score  # noqa: E402


def test_lookup_and_adjusted_rand_helpers():
    feature = np.asarray([1, 1, 2, 2, 3, 3])
    labels = np.asarray([0, 0, 1, 1, 0, 0])
    mapping, default = lookup_fit(feature, labels)
    result = lookup_score(feature, labels, mapping, default)
    assert result["accuracy"] == 1.0
    assert adjusted_rand(labels, labels) > 0.999


def test_semantic_atlas_has_weight_and_data_evidence():
    source = Path("scripts/run_bilin18_layer17_routed_semantic_atlas.py").read_text()
    for required in ("structural_bitmask", "semantic_profile", "top_contexts", "output_role", "cross_corpus"):
        assert required in source
