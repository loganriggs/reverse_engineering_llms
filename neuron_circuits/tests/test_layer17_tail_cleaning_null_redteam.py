import json
from pathlib import Path


def test_high_resolution_null_redteam_preserves_multiple_null_families():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_cleaning_null_redteam.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer17_tail_cleaning_null_redteam.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["null_replicates"] >= 128
    assert config["rotation_replicates"] >= 128
    for required in (
        "sign_null", "pairing_null", "rotation_null", "empirical_p",
        "top4_energy", "maximum_eigenvalue",
    ):
        assert required in source
