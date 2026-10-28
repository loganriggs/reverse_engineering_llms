import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_head_ranking_stability_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_head_ranking_stability.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["edge_budget"] == 64
    assert config["rediscovery_start"] + config["rediscovery_rows"] <= config["evaluation_start"]

