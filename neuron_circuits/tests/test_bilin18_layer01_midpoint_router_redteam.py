import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin18_layer01_midpoint_router_redteam import clustered_difference


def test_clustered_difference_wraps_pair_bootstrap():
    rows = [
        {"pair": "a", "left": 2.0, "right": 1.0},
        {"pair": "a", "left": 4.0, "right": 1.0},
        {"pair": "b", "left": 3.0, "right": 1.0},
    ]
    config = {"bootstrap_samples": 1000, "seed": 3}
    center, interval = clustered_difference(rows, "left", "right", config, 0)
    assert center == 2.0
    assert interval[0] <= center <= interval[1]
