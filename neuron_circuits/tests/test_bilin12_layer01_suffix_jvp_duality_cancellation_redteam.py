import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_suffix_jvp_duality_cancellation_redteam import duality_metrics


def test_cauchy_metric_handles_zero_dot_product():
    jvp = torch.tensor([[1.0, 0.0]])
    cotangent = torch.tensor([[0.0, 1.0]])
    tangent = torch.tensor([[1.0, 0.0]])
    vjp = torch.tensor([[0.0, 1.0]])
    row = duality_metrics(jvp, cotangent, tangent, vjp)
    assert row["cauchy_normalized_error"] == 0.0
    assert row["dot_alignment"] == 0.0
