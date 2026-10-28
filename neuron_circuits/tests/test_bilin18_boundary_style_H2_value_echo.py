import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_H2_value_echo import project  # noqa: E402


def test_value_echo_is_locked_before_results():
    config = json.loads((ROOT / "configs/bilin18_boundary_style_H2_value_echo.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["ranks"] == [1, 4, 16]
    assert config["random_subspaces"] >= 8


def test_projection_is_exact_on_basis_and_zero_off_basis():
    basis = torch.eye(4)[:2]
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.testing.assert_close(project(values, basis), torch.tensor([[1.0, 2.0, 0.0, 0.0]]))

