import json
from pathlib import Path

import torch


def test_restricted_redteam_predeclares_full_rank_side_and_metric_controls():
    config = json.loads(Path("configs/bilin18_layer8_restricted_tensor_redteam.json").read_text())
    assert config["ranks"][-1] == 1152
    assert len(config["ridge_fractions"]) == 3
    source = Path("scripts/run_bilin18_layer8_restricted_tensor_redteam.py").read_text()
    for required in ("context_only", "output_only", "global_square", "ridge_summary", "joint_test_r2"):
        assert required in source


def test_tucker_parameter_count_matches_factor_shapes():
    import sys
    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_bilin18_layer8_restricted_tensor_redteam import tucker_parameters
    assert tucker_parameters(8, output=7, context=5, edit=3) == 7 * 8 + 5 * 8 + 3 * 8 * 8
