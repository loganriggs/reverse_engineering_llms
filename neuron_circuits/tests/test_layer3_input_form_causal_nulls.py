import json
from pathlib import Path


def test_input_form_redteam_uses_adversarial_nulls_scales_and_factor_control():
    config = json.loads(Path("configs/bilin18_layer3_input_form_causal_nulls.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["scales"][0] == 0.1
    assert config["null_candidates"] >= 4
    source = Path("scripts/run_bilin18_layer3_input_form_causal_nulls.py").read_text()
    assert "max(sign_candidates" in source
    assert "max(pairing_candidates" in source
    assert "factor_metric" in source
    assert "delta_energy" in source
    assert "evaluate_writes" in source
