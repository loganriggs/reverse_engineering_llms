import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_typed_attention_edge_specificity_redteam import closure, decode, encode, randomized_support


def test_encode_decode_roundtrip():
    values = np.asarray([encode(2, 3, 4, 7, 21), encode(8, 6, 20, 7, 21)])
    head, distance, category = decode(values, 7, 21)
    assert head.tolist() == [2, 8]
    assert distance.tolist() == [3, 6]
    assert category.tolist() == [4, 20]


def test_category_randomization_preserves_head_distance_counts():
    selected = np.asarray([encode(0, 1, 2, 3, 5), encode(0, 1, 3, 3, 5), encode(1, 2, 4, 3, 5)])
    random = randomized_support(selected, "category", 2, 3, 5, np.random.default_rng(4))
    h0, d0, _ = decode(selected, 3, 5); h1, d1, _ = decode(random, 3, 5)
    assert sorted(zip(h0, d0)) == sorted(zip(h1, d1))


def test_category_closure_drops_only_category_condition():
    selected = np.asarray([encode(1, 2, 3, 4, 5)])
    expanded = closure(selected, "category", 3, 4, 5)
    assert len(expanded) == 5
