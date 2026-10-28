import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_selector_protocol_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_terminal_edit_dynamic_selector.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["examples_per_label_per_split"] >= 40
    assert config["batch_size"] <= 4
