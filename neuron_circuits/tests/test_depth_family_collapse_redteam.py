import json
from pathlib import Path


def test_depth_family_collapse_redteam_tests_competing_explanations():
    config = json.loads(Path(
        "configs/bilin18_depth_family_collapse_redteam.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_depth_family_collapse_redteam.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["input_rank"] >= 32
    assert config["permutation_replicates"] >= 10000
    for required in (
        "output_family_overlap", "input_family_overlap", "rotation_invariant_metric",
        "leave_one_pair_out", "coordinate_accuracy", "permutation_p",
        "wrong_family", "pair_balanced", "evaluate_writes",
    ):
        assert required in source
