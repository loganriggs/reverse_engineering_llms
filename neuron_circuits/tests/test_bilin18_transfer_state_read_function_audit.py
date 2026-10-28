import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_read_function_audit import (  # noqa: E402
    factorized_project,
    make_read_bases,
)


def test_read_function_audit_is_locked_on_unused_rows_and_targets():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_read_function_audit.json").read_text())
    assert config["status"] == "locked-before-read-function-audit-on-new-contexts-and-targets"
    assert config["evaluation_index_low"] == 19000
    assert config["evaluation_index_high"] == 20000
    assert config["targets"] == [[0, 7], [4, 2], [5, 1]]


def test_read_bases_are_orthonormal_and_causal():
    bases = make_read_bases(128, [31, 63, 95, 127], "cpu")
    for basis in bases.values():
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-6, rtol=1e-6)
    assert bases["current_sites4"].shape == (128, 4)
    assert bases["current_previous8"].shape == (128, 8)
    assert bases["recent4x4"].shape == (128, 16)
    assert bases["recent8x4"].shape == (128, 32)


def test_factorized_projection_uses_position_and_residual_spans():
    tangent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    positions = torch.eye(4)[:, :2]
    residuals = torch.eye(4)[:, :2]
    expected = torch.zeros_like(tangent)
    expected[:, :, :2, :2] = tangent[:, :, :2, :2]
    torch.testing.assert_close(factorized_project(tangent, positions, residuals), expected)
