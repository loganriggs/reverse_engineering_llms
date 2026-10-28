import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_position_head_sparsity_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_cross_position_head_sparsity.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["head_edge_ranks"][-1] == 72
    assert config["discovery_start"] + config["discovery_rows"] <= config["evaluation_start"]
    assert config["discovery_edit_position"] not in (63, 119)

