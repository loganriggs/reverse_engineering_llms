import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_path_decomposition import additive_effect  # noqa: E402


def test_additive_effect_sums_two_native_centered_paths():
    native = torch.tensor([1.0, 2.0])
    direct = torch.tensor([3.0, 1.0])
    attention = torch.tensor([0.0, 6.0])
    assert torch.equal(additive_effect(native, direct, attention), torch.tensor([1.0, 3.0]))
