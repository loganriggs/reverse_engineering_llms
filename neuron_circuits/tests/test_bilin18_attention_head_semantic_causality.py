import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_attention_head_semantic_causality import effect_size, selected_sites


def test_selected_sites_is_balanced_when_both_classes_are_available():
    raw = np.zeros((10, 5), dtype=np.int64)
    raw[:5, 2] = 1
    flags = np.asarray([0, 1], dtype=np.uint8)
    rows, positions, labels = selected_sites(raw, [1], flags, 4, 3)
    assert len(rows) == len(positions) == len(labels) == 8
    assert labels.sum() == 4


def test_effect_size_uses_calibrated_direction():
    labels = np.asarray([0, 0, 1, 1], dtype=bool)
    values = np.asarray([-2.0, -1.0, 1.0, 2.0])
    assert effect_size(values, labels, 1) > 1
    assert effect_size(values, labels, -1) < -1
