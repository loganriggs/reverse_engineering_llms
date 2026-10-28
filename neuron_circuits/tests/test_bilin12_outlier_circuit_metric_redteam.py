import json
from pathlib import Path


def test_metric_redteam_is_locked():
    config = json.loads(Path("configs/bilin12_outlier_circuit_metric_redteam.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["phases"]["late"] == [9, 10, 11]
    assert len(config["hypotheses"]) == 6


def test_metric_redteam_checks_aggregation_and_depth():
    source = Path("scripts/run_bilin12_outlier_circuit_metric_redteam.py").read_text()
    assert "aggregation_correlations" in source
    assert "leave_one_out" in source
    assert "unanimous_sign_top1_layers" in source
    assert "phase_summary" in source
