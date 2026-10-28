import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_degree_pilot_is_locked_and_complete():
    config = json.loads((ROOT / "configs/bilin18_layer1_suffix_degree_pilot.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["degrees"][-1] == 32
    assert config["sites_per_feature"] == 2

