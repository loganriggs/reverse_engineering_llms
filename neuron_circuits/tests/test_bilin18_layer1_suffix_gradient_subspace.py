import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suffix_gradient_subspace_is_locked_to_discovery_rank():
    config = json.loads((ROOT / "configs/bilin18_layer1_suffix_gradient_subspace.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["ranks"][-1] == 32
    assert config["support_rank"] == 64

