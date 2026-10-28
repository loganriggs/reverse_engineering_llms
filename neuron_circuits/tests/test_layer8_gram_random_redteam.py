import json
from pathlib import Path


def test_random_redteam_uses_both_query_families():
    config = json.loads(Path("configs/bilin18_layer8_gram_random_redteam.json").read_text())
    assert config["queries_per_family"] >= 4
    assert config["ranks"] == [4, 16]
