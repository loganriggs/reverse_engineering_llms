import json
from pathlib import Path


def test_boundary_code_config_is_locked():
    config = json.loads(Path("configs/bilin12_layer11_boundary_code_audit.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layer"] == 11
    assert config["address_count"] == 4
    assert len(config["hypotheses"]) == 8


def test_boundary_code_script_distinguishes_semantic_and_geometric_sharing():
    source = Path("scripts/run_bilin12_layer11_boundary_code_audit.py").read_text()
    assert "activation_pc_fraction" in source
    assert "shared_input_energy_fraction" in source
    assert "shared_output_energy_fraction" in source
    assert "position_demeaned_boundary_auc" in source
    assert "residual_correlation" in source
    assert "native_top32_jaccard" in source
    assert 'gram_a["condition"]' in source
