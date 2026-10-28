import json
from pathlib import Path


def test_natural_path_config_is_locked():
    config = json.loads(Path("configs/bilin12_layer11_boundary_natural_path_redteam.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layer"] == 11
    assert config["position"] == 127
    assert config["random_replicates"] == 8
    assert len(config["hypotheses"]) == 8


def test_natural_path_script_has_required_controls():
    source = Path("scripts/run_bilin12_layer11_boundary_natural_path_redteam.py").read_text()
    assert "address_span" in source
    assert "address_backbone" in source
    assert "state_pca4" in source
    assert "boundary_pca4" in source
    assert "random4" in source
    assert "state_energy_fraction" in source
    assert "write_delta_r2" in source
    assert "relative_prediction_kl" in source
