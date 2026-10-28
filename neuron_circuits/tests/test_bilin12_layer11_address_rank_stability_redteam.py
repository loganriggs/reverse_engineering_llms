import json
from pathlib import Path


def test_rank_stability_config_is_locked():
    config = json.loads(Path("configs/bilin12_layer11_address_rank_stability_redteam.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["ranks"] == [1, 2, 3, 4]
    assert config["layer"] == 11
    assert len(config["hypotheses"]) == 8


def test_rank_stability_script_compares_two_slices_and_controls():
    source = Path("scripts/run_bilin12_layer11_address_rank_stability_redteam.py").read_text()
    assert "address_a_rank" in source
    assert "address_b_rank" in source
    assert "state_pca_rank" in source
    assert "boundary_pca_rank" in source
    assert "cross_slice" in source
    assert "write_delta_r2" in source
    assert "relative_prediction_kl" in source
