import json
from pathlib import Path


def test_late_program_config_is_locked():
    config = json.loads(Path("configs/bilin12_late_address_program_audit.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layers"] == [10, 11]
    assert config["address_count"] == 4
    assert config["native_prefixes"][-1] == 128
    assert config["edit_scales"] == [0.25, 1.0]
    assert len(config["hypotheses"]) == 8


def test_late_program_script_has_algebra_semantics_and_causality():
    source = Path("scripts/run_bilin12_late_address_program_audit.py").read_text()
    assert "operator_k90" in source
    assert "random_direction" in source
    assert "recurrent_features" in source
    assert "full_linear" in source
    assert "rank1" in source
    assert "top128" in source
    assert "relative_prediction_kl" in source
