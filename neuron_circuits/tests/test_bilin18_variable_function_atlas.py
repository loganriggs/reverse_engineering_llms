import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_variable_function_atlas import effective_support, factored_frobenius_cosines


def test_effective_support_and_factored_cosine_match_explicit_maps():
    assert effective_support([1, 1, 0]) == 2
    generator = torch.Generator().manual_seed(4)
    factors = [(torch.randn(5, 2, generator=generator), torch.randn(2, 5, generator=generator)) for _ in range(3)]
    cosine, _ = factored_frobenius_cosines(factors)
    maps = [output @ value for output, value in factors]
    expected = np.asarray([[float(torch.sum(a * b) / (a.norm() * b.norm())) for b in maps] for a in maps])
    assert np.allclose(cosine, expected, atol=1e-6)
