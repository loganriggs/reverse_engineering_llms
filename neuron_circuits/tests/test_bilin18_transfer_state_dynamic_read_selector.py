import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_dynamic_read_selector import (  # noqa: E402
    lag_to_next_query,
    position_demand,
    select_sites,
)


def test_dynamic_selector_is_locked_after_invocation_gap():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_dynamic_read_selector.json").read_text())
    assert config["status"] == "locked-after-invocation-specific-early-gap-before-dynamic-site-selection"
    assert config["site_counts"] == [4, 8, 16, 32]


def test_position_demand_and_selection_are_weighted():
    adjoint = torch.zeros(2, 1, 3, 2)
    adjoint[0, 0, 1, 0] = 2
    adjoint[1, 0, 2, 0] = 3
    weights = torch.tensor([10., 1.])
    torch.testing.assert_close(position_demand(adjoint, weights), torch.tensor([0., 40., 9.], dtype=torch.float64))
    assert select_sites(adjoint, weights, 2) == [1, 2]


def test_lag_is_relative_to_next_scored_position():
    assert lag_to_next_query(30, [31, 63, 95, 127]) == 1
    assert lag_to_next_query(32, [31, 63, 95, 127]) == 31
