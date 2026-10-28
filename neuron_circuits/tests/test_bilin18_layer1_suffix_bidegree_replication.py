import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_bidegree_replication_is_locked():
    config = json.loads((ROOT / "configs/bilin18_layer1_suffix_bidegree_replication.json").read_text())
    assert config["status"].startswith("locked-")
    source = json.loads((ROOT / config["source_results"]).read_text())
    assert len(source["records"]) == 96

