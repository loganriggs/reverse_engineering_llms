import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_dynamic_alignment_replication import (
    first_threshold,
    pair_cluster_interval,
)
from run_bilin18_layer01_native_rotation_tournament import (
    inverse_rotate,
    make_rotation,
    rotate,
)


def test_rotation_round_trip_for_3072_channels():
    torch.manual_seed(0)
    value = torch.randn(3, 3072)
    permutations = make_rotation(3072, 5, 11, "cpu")
    assert torch.allclose(inverse_rotate(rotate(value, permutations), permutations), value, atol=2e-6, rtol=2e-6)


def test_threshold_and_pair_cluster_bootstrap_collapse_contexts():
    assert first_threshold([0.2, 0.6, 0.95], [32, 64, 128], 0.9) == 128
    assert first_threshold([0.2, 0.6], [32, 64], 0.9) is None
    records = [
        {"pair": "a", "value": 1.0}, {"pair": "a", "value": 1.0},
        {"pair": "b", "value": 2.0}, {"pair": "b", "value": 2.0},
        {"pair": "c", "value": 3.0}, {"pair": "c", "value": 3.0},
    ]
    estimate, interval = pair_cluster_interval(records, lambda row: row["value"], 1000, 0)
    assert estimate == 2.0
    assert np.isfinite(interval).all() and interval[0] <= estimate <= interval[1]
