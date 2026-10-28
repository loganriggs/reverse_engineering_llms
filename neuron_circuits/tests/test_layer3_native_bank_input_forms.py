import json
from pathlib import Path


def test_native_bank_input_forms_have_invariant_metric_nulls_and_causal_controls():
    config = json.loads(Path("configs/bilin18_layer3_native_bank_input_forms.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["output_rank"] == 4
    assert config["input_ranks"] == [4, 16, 32, 64]
    source = Path("scripts/run_bilin18_layer3_native_bank_input_forms.py").read_text()
    assert "metric += form @ form" in source
    assert "sign_null" in source
    assert "pairing_null" in source
    for label in ("correct", "global", "wrong", "low", "random"):
        assert f'"{label}"' in source
    assert "evaluate_writes" in source
