import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_tail_routed_program_controls import canonical_overlap_matrix  # noqa: E402


def test_overlap_matrix_is_rotation_invariant_on_diagonal():
    generator = torch.Generator().manual_seed(8)
    q = torch.linalg.qr(torch.randn(3, 12, 4, generator=generator)).Q
    result = canonical_overlap_matrix(q)
    assert result.shape == (3, 3)
    assert torch.allclose(torch.diag(result), torch.ones(3), atol=1e-5)


def test_route_controls_are_locked_and_matched():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_routed_program_controls.json"
    ).read_text())
    source = Path("scripts/run_bilin18_layer17_tail_routed_program_controls.py").read_text()
    assert "locked" in config["status"]
    for required in (
        "native_hard", "shuffled_hard", "position_mean", "position_hard",
        "uniform", "fixed_{i}", "global32", "canonical_overlap_matrix",
    ):
        assert required in source
