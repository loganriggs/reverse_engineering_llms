from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_semantic_group_null import (  # noqa: E402
    evaluate_grouping,
    group_members,
    top_projection,
)


def test_group_members_excludes_target():
    labels = np.array([0, 0, 1, 1, 1])
    assert group_members(labels, 0).tolist() == [1]
    assert group_members(labels, 3).tolist() == [2, 4]


def test_projection_and_grouping_use_discovery_only():
    discovery = np.array([[4.0, 1.0], [3.0, 2.0], [0.0, 5.0], [1.0, 4.0]])
    evaluation = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    assert top_projection(discovery[1], evaluation[0], 1) == 2.0
    assert evaluate_grouping(labels, discovery, evaluation, [1])[1] == 2.0
