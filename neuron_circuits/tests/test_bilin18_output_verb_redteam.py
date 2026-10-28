import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_output_verb_redteam import median_nearest


def test_median_nearest_uses_absolute_cosine_and_excludes_self():
    vectors = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    value, cosine = median_nearest(vectors)
    assert value == 1.0
    assert cosine[0, 0] == -1
