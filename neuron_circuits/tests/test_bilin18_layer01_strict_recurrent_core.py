from pathlib import Path
import json


def test_config_declares_strict_core_threshold():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/bilin18_layer01_strict_recurrent_core.json").read_text())
    assert config["reference_rank"] == 256
    assert "S1_strict_core_captures_half_of_bank_effect" in config["hypotheses"]
