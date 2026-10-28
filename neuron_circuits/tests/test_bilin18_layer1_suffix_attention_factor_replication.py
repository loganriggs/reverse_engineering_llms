import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_attention_factor_replication_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_suffix_attention_factor_replication.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["attention_layers"] == list(range(2, 10))
    assert config["support_rank"] == 64

