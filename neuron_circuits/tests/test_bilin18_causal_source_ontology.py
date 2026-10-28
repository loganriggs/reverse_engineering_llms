import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_causal_source_ontology import normalize_profiles


def test_profile_normalization_removes_coordinate_scale_and_is_finite():
    sums = torch.tensor([[2.0, 8.0], [1.0, 2.0], [0.0, 0.0]])
    counts = torch.tensor([[2.0, 4.0], [1.0, 1.0], [0.0, 0.0]])
    result = normalize_profiles(sums, counts)
    assert result.shape == sums.shape
    assert torch.isfinite(result).all()
    assert torch.allclose(result[-1], torch.zeros(2))
