import json
from pathlib import Path


def test_attention_frozen_routes_use_disjoint_contexts_and_head_nulls():
    config = json.loads(
        Path("configs/bilin18_layer8_semantic_attention_frozen_routes.json").read_text()
    )
    assert config["discovery_context_index"] != config["evaluation_context_index"]
    assert config["ranks"] == [8, 16, 32, 64, 128]
    assert Path(config["rank_results"]).exists()
    source = Path("scripts/run_bilin18_layer8_semantic_attention_frozen_routes.py").read_text()
    assert "head_shift" in source
    assert "mean_absolute_projection" in source
    assert "oracle64" in source
