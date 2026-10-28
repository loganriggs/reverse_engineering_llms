import json
from pathlib import Path


def test_embedding_residual_causality_uses_thirty_two_worlds():
    config = json.loads(
        Path("configs/bilin18_layer4_embedding_residual_causality.json").read_text()
    )
    assert config["layer"] == 4
    assert Path(config["frozen_results"]).exists()
    source = Path("scripts/run_bilin18_layer4_embedding_residual_causality.py").read_text()
    assert "residual_state" in source
    assert "embedding_stream" in source
    assert "for factor_mask in range(32)" in source
    assert "construction_cancellation_ratio" in source
