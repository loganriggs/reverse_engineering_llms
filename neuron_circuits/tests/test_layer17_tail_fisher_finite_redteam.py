import json
from pathlib import Path


def test_finite_fisher_redteam_locks_local_and_full_doses():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_fisher_finite_redteam.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer17_tail_fisher_finite_redteam.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["dose_fractions"][0] <= 0.1
    assert config["dose_fractions"][-1] == 1.0
    assert "fisher_rank4" in config["variants"]
    assert "frobenius_rank4" in config["variants"]
    for required in (
        "predicted_kl", "exact_kl", "dose_squared", "calibration_ratio",
        "final_logit_fisher", "correlation",
    ):
        assert required in source
