import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_semantic_test_is_locked_and_sparse():
    config = json.loads((ROOT / "configs/bilin18_layer1_dynamic_semantic_coalitions.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["rank"] == 64
    assert "upper bound" in config["hypotheses"]["D5_dynamic_selector_is_an_upper_bound"]
