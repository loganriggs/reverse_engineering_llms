import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_output_verb_transplant import matched_controls


def test_matched_controls_have_exact_partner_angle():
    native = torch.tensor([1.0, 0.0, 0.0])
    partner = torch.tensor([0.5, 3 ** 0.5 / 2, 0.0])
    bank = torch.eye(3)
    aligned, controls, cosine = matched_controls(native, partner, bank, 5, 9)
    assert abs(float(torch.dot(aligned, native)) - cosine) < 1e-6
    assert torch.allclose(controls @ native, torch.full((5,), cosine), atol=1e-6)
