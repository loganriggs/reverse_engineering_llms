import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_terminal_semantic_source_factorial_is_locked():
    config = json.loads((ROOT / "configs/bilin18_terminal_semantic_source_factorial.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["examples_per_label_per_split"] >= 40
    assert config["batch_size"] <= 8
