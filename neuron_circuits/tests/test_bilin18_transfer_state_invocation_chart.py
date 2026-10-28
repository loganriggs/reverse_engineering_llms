import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_invocation_chart import (  # noqa: E402
    chart_specs,
    invocation_project,
)


def test_invocation_chart_is_locked_after_shared_result():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_invocation_chart.json").read_text())
    assert config["status"] == "locked-after-shared-read-gap-before-invocation-specific-chart"
    assert config["source_result"].endswith("bilin18_transfer_state_read_function_audit.json")


def test_chart_specs_have_exact_coordinate_budget():
    specs = chart_specs([31, 63, 95, 127], 128)
    assert {name: len(sites) * rank for name, (sites, rank) in specs.items()} == {
        "current_specific32": 128,
        "current_previous_specific16": 128,
        "recent4_specific8": 128,
        "recent8_specific4": 128,
    }


def test_invocation_projection_uses_local_basis_only():
    tangent = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    bases = {1: torch.eye(4)[:, :2]}
    expected = torch.zeros_like(tangent)
    expected[:, :, 1, :2] = tangent[:, :, 1, :2]
    torch.testing.assert_close(invocation_project(tangent, bases), expected)
