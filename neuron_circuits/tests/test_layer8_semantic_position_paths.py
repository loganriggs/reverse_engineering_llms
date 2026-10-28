import json
from pathlib import Path


def test_semantic_position_paths_cover_full_causal_suffix():
    config = json.loads(Path("configs/bilin18_layer8_semantic_position_paths.json").read_text())
    assert config["ranks"][-1] == 17
    assert 4 in config["ranks"] and 8 in config["ranks"]
    assert Path(config["source_config"]).exists()
