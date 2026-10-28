import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_pair_capability_protocol_is_locked():
    config = json.loads((ROOT / "configs/bilin18_cross_pair_latent_capability.json").read_text())
    families = json.loads((ROOT / config["family_config"]).read_text())["families"]
    assert config["status"].startswith("locked-")
    assert len(families) == 5
    assert config["random_contrasts"] >= 31
    assert config["examples_per_label_per_split"] >= 40
