import json
from pathlib import Path


def test_contrast_gain_config_is_locked():
    config = json.loads(Path("configs/bilin12_layer11_address_contrast_gain_audit.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["address_count"] == 4
    assert len(config["hypotheses"]) == 6


def test_contrast_gain_script_measures_redundancy_and_amplification():
    source = Path("scripts/run_bilin12_layer11_address_contrast_gain_audit.py").read_text()
    assert "address_energy_fraction" in source
    assert "operator_energy_fraction" in source
    assert "write_r2_increment" in source
    assert "state_energy_increment" in source
    assert "amplification_ratio" in source
    assert "spearman" in source
