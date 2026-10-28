import json
from pathlib import Path


def test_value_bus_causality_uses_sixteen_worlds_and_both_value_paths():
    config = json.loads(
        Path("configs/bilin18_layer8_semantic_attention_value_bus_causality.json").read_text()
    )
    assert config["route_rank"] == 64
    assert Path(config["factor_results"]).exists()
    source = Path("scripts/run_bilin18_layer8_semantic_attention_value_bus_causality.py").read_text()
    assert "current_value" in source
    assert "value_bus" in source
    assert "for factor_mask in range(16)" in source
    assert "cancellation_ratio" in source
