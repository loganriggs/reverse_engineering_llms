import json
from pathlib import Path


def test_rank_semantic_config_is_locked():
    config = json.loads(Path("configs/bilin12_late_program_rank_semantic_redteam.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["ranks"][:3] == [1, 2, 4]
    assert config["native_prefixes"][-1] == 512
    assert len(config["hypotheses"]) == 8


def test_rank_semantic_script_checks_stability_causality_and_edits():
    source = Path("scripts/run_bilin12_late_program_rank_semantic_redteam.py").read_text()
    assert "canonical_cosines" in source
    assert "best_permutation" in source
    assert "rank_frontier" in source
    assert "native_frontier" in source
    assert "newline_substitution" in source
    assert "predicted_direction_fraction" in source
