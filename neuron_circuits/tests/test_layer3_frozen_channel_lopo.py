import json
from pathlib import Path


def test_layer3_lopo_redteam_removes_pair_and_checks_all_wrong_families():
    config = json.loads(Path("configs/bilin18_layer3_frozen_channel_lopo.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    source = Path("scripts/run_bilin18_layer3_frozen_channel_lopo.py").read_text()
    assert "family_lopo" in source
    assert "global_lopo" in source
    assert "if candidate_pair != record[\"pair\"]" in source
    assert "wrong_family_median" in source
    assert "wrong_family_best" in source
