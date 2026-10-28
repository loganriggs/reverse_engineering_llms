from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_native_rotation_redteam import normalized_score, participation_ratio  # noqa: E402


def test_participation_ratio_extremes():
    assert participation_ratio(torch.tensor([1.0, 0.0, 0.0, 0.0])) == 1.0
    assert abs(participation_ratio(torch.ones(8)) - 8.0) < 1e-10


def test_normalized_score_removes_decoder_scale():
    decoder = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
    score = torch.tensor([1.0, 2.0, 4.0])
    assert torch.allclose(normalized_score(score, decoder), torch.ones(3))
