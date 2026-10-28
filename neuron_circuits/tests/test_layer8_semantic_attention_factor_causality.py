import json
from pathlib import Path


def test_attention_factor_causality_uses_exact_factorial_worlds():
    config = json.loads(
        Path("configs/bilin18_layer8_semantic_attention_factor_causality.json").read_text()
    )
    assert config["route_rank"] == 64
    assert config["layers"] == [3, 4, 5]
    assert Path(config["frozen_results"]).exists()
    source = Path("scripts/run_bilin18_layer8_semantic_attention_factor_causality.py").read_text()
    assert "factor_edge_write" in source
    assert "factor_shapley" in source
    assert "for mask in range(8)" in source
    assert "oracle64" in source
