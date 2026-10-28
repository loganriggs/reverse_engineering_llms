import json
from pathlib import Path


def test_cleaning_stability_sweeps_sample_size_and_ridge():
    config = json.loads(Path("configs/bilin12_cleaning_stability_redteam.json").read_text())
    assert config["discovery_state_counts"] == [1024, 4096, 8192]
    assert config["ridge_fractions"] == [0.0001, 0.01, 0.1]
    source = Path("scripts/run_bilin12_cleaning_stability_redteam.py").read_text()
    for required in ("evaluation_second_moment_mismatch", "median_natural_r2", "median_gaussian_r2"):
        assert required in source
