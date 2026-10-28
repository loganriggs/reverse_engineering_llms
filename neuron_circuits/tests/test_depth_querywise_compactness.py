import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_depth_querywise_compactness import form_summaries  # noqa: E402


def test_form_summary_sorts_positive_and_negative_modes_by_magnitude():
    matrices = torch.diag(torch.tensor([-10.0, -3.0, 2.0, 1.0, 0.1]))[None]
    config = {"top_mode_count": 1, "extended_mode_count": 2}
    result = form_summaries(matrices, config)
    total = 100 + 9 + 4 + 1 + 0.01
    assert abs(result["median_top4_energy"] - 100 / total) < 1e-6
    assert abs(result["median_top16_energy"] - 109 / total) < 1e-6
    assert result["median_rank90"] == 2
