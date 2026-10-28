import json
from pathlib import Path


def test_control_routes_are_distinct():
    config = json.loads(Path("configs/bilin18_layer8_frozen_fisher_controls.json").read_text())
    assert len(config["routes"]) == len(set(config["routes"]))
    assert "identity" in config["routes"]
    assert "shuffled_full" in config["routes"]
