from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_coordinate_frontier import format_threshold_range, threshold_rank  # noqa: E402


def test_threshold_rank_returns_first_crossing():
    summary = {
        "route_1": {"downstream_fidelity": .1},
        "route_2": {"downstream_fidelity": .6},
        "route_4": {"downstream_fidelity": .55},
    }
    assert threshold_rank(summary, "route", [1, 2, 4], .5) == 2
    assert threshold_rank(summary, "route", [1, 2, 4], .9) is None
    assert format_threshold_range([2048, 4096]) == "2048–4096"
    assert format_threshold_range([4096, 4096]) == "4096"
    assert format_threshold_range([4096, None]) == "4096–not reached"
