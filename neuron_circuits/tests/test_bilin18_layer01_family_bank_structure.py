from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_family_bank_structure import recurrence_ranking, sign_agreement  # noqa: E402


def test_recurrence_ranking_prefers_repeated_coordinates():
    rows = torch.tensor([[4.0, 3.0, 0.0, 0.0], [5.0, 0.0, 3.0, 0.0]])
    score, frequency = recurrence_ranking(rows, 2)
    assert frequency.tolist() == [2, 1, 1, 0]
    assert int(score.argmax()) == 0


def test_sign_agreement_is_one_for_shared_sign_and_zero_for_split():
    rows = torch.tensor([[2.0, 1.0], [3.0, -1.0]])
    assert sign_agreement(rows, torch.tensor([0])) == 1.0
    assert sign_agreement(rows, torch.tensor([1])) == 0.0
