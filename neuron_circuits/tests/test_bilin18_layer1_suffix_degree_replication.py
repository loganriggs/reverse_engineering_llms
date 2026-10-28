import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_degree_replication_is_locked_to_all_pairs():
    config = json.loads((ROOT / "configs/bilin18_layer1_suffix_degree_replication.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["degrees"][-1] == 24
    source = json.loads((ROOT / config["source_results"]).read_text())
    assert len(source["records"]) == 96

