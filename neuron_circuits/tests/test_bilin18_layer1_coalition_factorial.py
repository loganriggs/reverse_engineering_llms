import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_coalition_factorial_is_locked():
    config = json.loads((ROOT / "configs/bilin18_layer1_coalition_factorial.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["rank"] == 64
    assert config["gradient_sites_per_feature"] >= 4

