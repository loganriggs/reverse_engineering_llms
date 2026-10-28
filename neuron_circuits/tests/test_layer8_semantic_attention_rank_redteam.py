import json
from pathlib import Path


def test_attention_rank_redteam_is_locked_and_rank_resolved():
    config = json.loads(
        Path("configs/bilin18_layer8_semantic_attention_rank_redteam.json").read_text()
    )
    assert config["top_ks"] == [1, 2, 4, 8, 16, 32, 64]
    assert config["rank_bands"] == [[0, 8], [8, 16], [16, 32], [32, 64]]
    assert config["layers"] == [3, 4, 5]
    assert Path(config["causal_results"]).exists()
    source = Path("scripts/run_bilin18_layer8_semantic_attention_rank_redteam.py").read_text()
    assert "all_edge_projections" in source
    assert "edited_source_top8" in source
    assert "nonedited_source_top8" in source
