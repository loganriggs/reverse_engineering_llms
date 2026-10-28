import json
from pathlib import Path


def test_conditioned_spike_redteam_has_matched_controls():
    config = json.loads(Path(
        "configs/bilin18_layer3_conditioned_spike_redteam.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_layer3_conditioned_spike_redteam.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["null_replicates"] >= 64
    assert config["random_subspaces"] >= 64
    for required in (
        "sign_null", "pairing_null", "random_subspace", "canonicalize",
        "empirical_p", "tail_top4", "random_tail_95",
    ):
        assert required in source
