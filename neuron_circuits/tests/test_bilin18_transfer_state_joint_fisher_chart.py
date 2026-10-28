import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_joint_fisher_chart import joint_project


def test_joint_chart_is_locked_on_new_targets():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_joint_fisher_chart.json").read_text())
    assert config["status"] == "locked-before-joint-position-residual-chart-on-new-targets"
    assert config["targets"] == [[1, 6], [3, 4], [5, 7]]


def test_joint_projection_preserves_shape_and_selected_field():
    tangent = torch.tensor([[[[1., 2.], [3., 4.]]]])
    axis = torch.tensor([[1.], [0.], [0.], [0.]])
    projected = joint_project(tangent, axis)
    torch.testing.assert_close(projected, torch.tensor([[[[1., 0.], [0., 0.]]]]))
