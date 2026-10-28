import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_attention_factor_pilot_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_suffix_attention_factor_pilot.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["attention_layers"] == list(range(2, 10))
    assert config["sites_per_feature"] == 2
    assert config["support_rank"] == 64

