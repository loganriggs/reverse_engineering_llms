import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer17_tail_circuit_redteam import (  # noqa: E402
    axis_alignment_statistic,
    cumulative_subspace_coverage,
)


def test_cumulative_coverage_finds_delayed_target_axes():
    basis = np.eye(8)
    target = basis[:, 4:6]
    rows = cumulative_subspace_coverage(target, basis, [2, 4, 6, 8])
    assert rows == {"2": 0.0, "4": 0.0, "6": 1.0, "8": 1.0}


def test_axis_alignment_changes_under_nontrivial_rotation():
    scores = np.asarray([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    flags = np.asarray([[1], [1], [0], [0]], dtype=np.uint8)
    identity = axis_alignment_statistic(scores, flags)
    rotation = np.asarray([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2)
    rotated = axis_alignment_statistic(scores @ rotation, flags)
    assert not np.isclose(identity, rotated)


def test_tail_redteam_is_locked_and_uses_fair_rotation_null():
    config = json.loads(Path("configs/bilin18_layer17_tail_circuit_redteam.json").read_text())
    source = Path("scripts/run_bilin18_layer17_tail_circuit_redteam.py").read_text()
    assert "locked" in config["status"]
    assert config["rotation_null_replicates"] >= 64
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    for required in (
        "weighted_output", "cumulative_subspace_coverage", "rotation_null",
        "top4_form_energy", "prediction_correlation", "median_kl",
    ):
        assert required in source
