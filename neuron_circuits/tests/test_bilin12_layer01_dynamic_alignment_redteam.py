import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_dynamic_alignment_redteam import jaccard


def test_jaccard_has_expected_support_semantics():
    assert jaccard(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])) == 0.5
    assert jaccard(torch.tensor([1, 2]), torch.tensor([1, 2])) == 1.0
