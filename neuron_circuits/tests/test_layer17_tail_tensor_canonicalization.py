import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_tail_tensor_canonicalization import (  # noqa: E402
    best_axis_match,
    centered_vector_r2,
)


def test_axis_match_is_invariant_to_permutation_and_sign():
    first = np.eye(4)
    second = first[:, [2, 0, 3, 1]] * np.asarray([-1, 1, -1, 1])
    match = best_axis_match(first, second)
    assert np.isclose(match["mean_cosine"], 1)
    assert np.isclose(match["minimum_cosine"], 1)


def test_centered_vector_r2_ignores_constant_offset():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(100, 4))
    prediction = target + np.asarray([1, 2, 3, 4])
    assert np.isclose(centered_vector_r2(target, prediction), 1)


def test_tensor_canonicalization_is_locked_and_basis_invariant():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_tensor_canonicalization.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer17_tail_tensor_canonicalization.py"
    ).read_text()
    assert "locked" in config["status"]
    assert 16 in config["joint_ranks"]
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    for required in (
        "tensor_gram", "canonical_output", "common_input", "joint_energy",
        "diagonal_only", "behavioral", "native_tail_score",
    ):
        assert required in source
