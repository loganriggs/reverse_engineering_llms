import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_radius_frontier_is_locked_and_interpolates():
    config = json.loads((ROOT / "configs/bilin12_layer01_edit_radius_frontier.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["scales"] == [0.25, 0.5, 0.75, 1.0]
    assert config["sites_per_corpus"] >= 32
    assert config["rank"] == 256
