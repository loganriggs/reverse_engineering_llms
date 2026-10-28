import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_suffix_fisher_chart_router import repeat_first


def test_repeat_first_only_repeats_batch_axis():
    value = torch.arange(6).reshape(1, 2, 3)
    repeated = repeat_first(value, 4)
    assert repeated.shape == (4, 2, 3)
    assert torch.equal(repeated[3], value[0])
