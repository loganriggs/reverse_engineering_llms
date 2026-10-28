import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_query_mode_sparsity import support_statistics  # noqa: E402


def test_support_statistics_handles_signed_sparse_contributions():
    contributions = torch.tensor([[[9.0, -1.0, 0.0, 0.0]], [[0.0, 0.0, 8.0, -2.0]]])
    result = support_statistics(contributions, occupancy_rank=1)
    assert result["median_k90_absolute_mass"] == 1
    assert result["median_effective_support"] > 1
    assert result["median_cancellation_ratio"] < 1
    assert result["median_occupancy_union_modes"] == 2
