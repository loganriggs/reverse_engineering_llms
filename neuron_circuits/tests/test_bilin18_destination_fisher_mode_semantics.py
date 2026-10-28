import json
from pathlib import Path


def test_semantic_panel_is_later_than_fisher_panel():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_destination_fisher_mode_semantics.json").read_text())
    assert config["evaluation_start"]>=19948
    assert config["evaluation_rows"]==32
