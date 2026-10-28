import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path("scripts").resolve()))
from bilin_checkpoint_utils import final_logits  # noqa: E402


def test_independent_natural_validation_has_usage_cleaning_and_causal_levels():
    source = Path("scripts/run_bilin12_natural_outlier_validation.py").read_text()
    for required in ("variance_capture", "ridge_covariance_transforms", "reconstruct_quadratic", "final_logits", "exact_kl", "random4"):
        assert required in source


def test_checkpoint_utility_projection_formula_is_present():
    source = Path("scripts/bilin_checkpoint_utils.py").read_text()
    assert "mean + (centered @ basis) @ basis.T" in source


def test_causal_projection_mean_includes_down_bias():
    source = Path("scripts/run_bilin12_natural_outlier_validation.py").read_text()
    assert "causal_mean = mean + block.mlp.Down_bias.float()" in source
    assert "final_logits(model, ids, layer, -1, causal_mean, basis)" in source
