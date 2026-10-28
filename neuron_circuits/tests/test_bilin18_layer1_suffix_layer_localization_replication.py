import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_layer_localization_replication_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_suffix_layer_localization_replication.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["window_sizes"] == [1, 2, 4, 8, 12, 16]
    assert config["support_rank"] == 64

