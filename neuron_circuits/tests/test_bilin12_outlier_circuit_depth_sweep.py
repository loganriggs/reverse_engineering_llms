import json
from pathlib import Path


def test_depth_sweep_is_locked_and_complete():
    config = json.loads(Path("configs/bilin12_outlier_circuit_depth_sweep.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layers"] == list(range(12))
    assert config["address_count"] == 4
    assert config["sign_null_count"] == 4
    assert config["covariance_ridge_fraction"] == 0.1
    assert len(config["hypotheses"]) == 8


def test_depth_sweep_measures_distinct_notions():
    source = Path("scripts/run_bilin12_outlier_circuit_depth_sweep.py").read_text()
    assert "leading_log_excess" in source
    assert "natural_rank1_efficiency" in source
    assert "operator_k90" in source
    assert "shapley_k90" in source
    assert "top8_recurrence" in source
    assert "output_role_fraction" in source
    assert "sign_null" in source
