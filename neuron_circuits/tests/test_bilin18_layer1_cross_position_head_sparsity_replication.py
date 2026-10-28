import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_position_head_sparsity_replication_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_cross_position_head_sparsity_replication.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["head_edge_ranks"][-1] == 144
    assert 72 in config["head_edge_ranks"]
    assert config["evaluation_start"] > 8500

