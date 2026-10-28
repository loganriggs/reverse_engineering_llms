import json
from pathlib import Path


def test_residual_telescope_null_preserves_targets_and_crosses_families():
    config = json.loads(Path("configs/bilin18_layer4_residual_telescope_null.json").read_text())
    assert Path(config["semantic_results"]).exists()
    source = Path("scripts/run_bilin18_layer4_residual_telescope_null.py").read_text()
    assert "deranged_targets" in source
    assert "target_multiset_preserved" in source
    assert "cross_family" in source
    assert "profile_cosine" in source
