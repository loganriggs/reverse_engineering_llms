import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervised_causal_test_is_locked_and_diverse():
    config = json.loads((ROOT / "configs/bilin18_layer1_supervised_coalition_causality.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layer"] == 1
    assert config["rank"] == 64
    assert len(config["features"]) == 8
    assert config["extreme_pairs"] * 2 == config["pairs_per_feature"]

