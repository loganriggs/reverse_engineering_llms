import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin18_layer8_joint_fisher_scale_redteam import run_routes  # noqa: E402,F401


def test_scale_grid_key_roundtrip():
    scales = [0.01, 0.05, 0.1, 0.2]
    payload = {str(scale): scale for scale in scales}
    assert [payload[str(scale)] for scale in scales] == scales
