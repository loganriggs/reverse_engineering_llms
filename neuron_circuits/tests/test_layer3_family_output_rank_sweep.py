import json
from pathlib import Path


def test_family_output_rank_sweep_has_lopo_global_wrong_and_native_controls():
    config = json.loads(Path("configs/bilin18_layer3_family_output_rank_sweep.json").read_text())
    assert config["output_ranks"] == [1, 2, 4, 8, 16, 32]
    source = Path("scripts/run_bilin18_layer3_family_output_rank_sweep.py").read_text()
    assert "candidate_pair != pair" in source
    assert "global_" in source
    assert "wrong_best_" in source
    assert "native_lopo" in source
    assert "evaluate_writes" in source
