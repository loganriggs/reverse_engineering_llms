import json
from pathlib import Path


def test_layer3_powerlaw_test_has_locked_nulls_and_strict_spike_gates():
    config = json.loads(Path("configs/bilin18_layer3_powerlaw_spikes.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["layer"] == 3
    assert config["null_replicates"] >= 16
    source = Path("scripts/run_bilin18_layer3_powerlaw_spikes.py").read_text()
    assert "sign_nulls" in source
    assert "pairing_nulls" in source
    assert "top_log_excess" in source
    assert "symmetric_channel_gram" in source
