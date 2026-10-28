import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stratified_gradient_replication_is_locked():
    config = json.loads((ROOT / "configs/bilin18_layer1_stratified_gradient_subspace.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["ranks"][-1] == 256
    assert config["support_rank"] == 64

