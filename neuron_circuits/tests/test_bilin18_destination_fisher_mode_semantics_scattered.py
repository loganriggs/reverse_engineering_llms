import json
from pathlib import Path


def test_scattered_semantic_panel_uses_random_rows():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/"configs/bilin18_destination_fisher_mode_semantics_scattered.json").read_text())
    assert config["evaluation_rows"]==64
    assert config["evaluation_index_high"]<=19000
    assert "evaluation_random_seed" in config
