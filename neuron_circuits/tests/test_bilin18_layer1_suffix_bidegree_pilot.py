import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_bidegree_pilot_is_locked():
    config = json.loads((ROOT / "configs/bilin18_layer1_suffix_bidegree_pilot.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["maximum_total_degree"] == 24
    assert config["sites_per_feature"] == 2

