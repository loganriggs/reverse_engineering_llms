import json
from pathlib import Path


def test_native_bank_svd_has_weight_activation_bottom_and_random_controls():
    config = json.loads(Path("configs/bilin18_layer3_native_bank_output_svd.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["ranks"][-1] == 32
    source = Path("scripts/run_bilin18_layer3_native_bank_output_svd.py").read_text()
    for label in ("weight_top", "weight_bottom", "activation", "random"):
        assert f'"{label}"' in source
    assert "candidate_pair != pair" in source
    assert "evaluate_writes" in source
