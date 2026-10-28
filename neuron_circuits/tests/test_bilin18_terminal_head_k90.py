import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_terminal_k90_protocol_is_locked():
    config = json.loads((ROOT / "configs/bilin18_terminal_head_k90.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["fidelity_threshold"] == 0.9
    assert config["examples_per_label_per_split"] >= 40
