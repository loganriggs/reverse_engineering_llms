import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_output_aware_test_is_locked_to_existing_pairs():
    config = json.loads((ROOT / "configs/bilin18_layer1_output_aware_coalitions.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["gradient_sites_per_feature"] >= 4
    source = json.loads((ROOT / config["source_config"]).read_text())
    assert source["rank"] == 64

