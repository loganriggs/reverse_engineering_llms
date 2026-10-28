import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_weighted_decomposition_is_locked():
    config = json.loads((ROOT / "configs/bilin12_layer01_weighted_coalition_decomposition.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["sites_per_corpus"] >= 64
    assert config["rank"] == 256
