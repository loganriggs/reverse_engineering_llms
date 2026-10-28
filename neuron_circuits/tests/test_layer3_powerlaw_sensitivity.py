import json
from pathlib import Path


def test_layer3_powerlaw_redteam_has_window_and_direct_null_tests():
    config = json.loads(Path("configs/bilin18_layer3_powerlaw_sensitivity.json").read_text())
    assert len(config["bulk_starts"]) >= 5
    assert len(config["bulk_stops"]) >= 3
    source = Path("scripts/run_bilin18_layer3_powerlaw_sensitivity.py").read_text()
    assert "empirical_null" in source
    assert "exponential" in source
    assert "heldout" in source
    assert "fit_powerlaw" in source
