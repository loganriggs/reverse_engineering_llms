import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_supervised_feature_census import FEATURES
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, distance_ids, edge_type_ids, token_categories


def test_token_categories_are_exclusive_and_priority_ordered():
    flags = np.zeros((3, len(FEATURES)), dtype=np.uint8)
    flags[0, FEATURES.index("newline")] = 1; flags[0, FEATURES.index("punctuation")] = 1
    flags[1, FEATURES.index("numeric")] = 1
    categories = token_categories(flags)
    assert categories.tolist() == [CATEGORIES.index("newline"), CATEGORIES.index("numeric"), CATEGORIES.index("other")]


def test_edge_type_ids_assign_one_type_per_causal_edge():
    category = torch.tensor([[1, 2, 3, 4]])
    distance = distance_ids([1, 3], 4, [0, 1, 2, 4], "cpu")
    ids = edge_type_ids(category, distance, 2, 5)
    assert ids.shape == (1, 2, 2, 4)
    assert int((ids >= 0).sum()) == 2 * (2 + 4)
    assert int(ids[ids >= 0].max()) < 2 * 3 * 5
