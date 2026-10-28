import json
from pathlib import Path


def test_semantic_path_decomposition_is_full_factorial():
    config = json.loads(Path("configs/bilin18_layer8_semantic_path_decomposition.json").read_text())
    assert config["contexts_per_pair"] >= 3
    assert len(config["hypotheses"]) == 6
    assert Path(config["source_config"]).exists()
