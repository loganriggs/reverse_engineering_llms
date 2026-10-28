import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_vocabulary import canonical_stability, retained_energy


def test_state_vocabulary_is_prospectively_locked():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_vocabulary.json").read_text())
    assert config["status"] == "locked-after-transfer-recurrence-validation-before-state-vocabulary-test"
    assert config["boundaries"] == [6, 10, 14, 17]
    assert config["rank"] == 64


def test_subspace_metrics_are_basis_invariant():
    basis = torch.eye(4, dtype=torch.float64)[:, :2]
    rotated = basis @ torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([4., 2., 1., 1.], dtype=torch.float64))
    assert retained_energy(covariance, basis) == retained_energy(covariance, rotated)
    score, values = canonical_stability(basis, rotated)
    assert abs(score - 1) < 1e-12
    assert all(abs(value - 1) < 1e-12 for value in values)
