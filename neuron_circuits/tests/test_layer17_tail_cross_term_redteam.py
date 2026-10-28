import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_tail_cross_term_redteam import program_prediction  # noqa: E402


def test_block_program_suppresses_cross_terms():
    z = torch.tensor([[1.0, 2.0]])
    forms = torch.tensor([[[1.0, 3.0], [3.0, 2.0]]])
    q1, q2 = torch.eye(2)[:, :1], torch.eye(2)[:, 1:]
    full = program_prediction(z, forms, [torch.eye(2)])
    block = program_prediction(z, forms, [q1, q2])
    assert torch.allclose(full, torch.tensor([[21.0]]))
    assert torch.allclose(block, torch.tensor([[9.0]]))


def test_cross_term_redteam_is_locked():
    config = json.loads(Path("configs/bilin18_layer17_tail_cross_term_redteam.json").read_text())
    source = Path("scripts/run_bilin18_layer17_tail_cross_term_redteam.py").read_text()
    assert "locked" in config["status"]
    for required in ("full_union16", "block_union16", "cross_terms", "exact_kl", "ce_delta"):
        assert required in source
