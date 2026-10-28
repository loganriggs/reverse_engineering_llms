import json
from pathlib import Path


def test_attention_atlas_uses_exact_three_factor_decomposition():
    config = json.loads(Path("configs/bilin18_layer8_semantic_attention_atlas.json").read_text())
    assert config["contexts_per_pair"] >= 2
    assert config["top_edges_per_layer"] >= 8
    assert len(config["hypotheses"]) == 6
