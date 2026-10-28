import json
from pathlib import Path


def test_mass_redteam_uses_new_split_and_registered_controls():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/bilin18_typed_attention_edge_mass_redteam.json").read_text())
    assert config["evaluation_start"] == 19200
    assert config["status"].startswith("locked-after-joint-binding")
    assert config["methods"] == ["operator128", "frequency128", "routing_mass128", "transport_mass128"]
