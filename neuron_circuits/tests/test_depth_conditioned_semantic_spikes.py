import json
from pathlib import Path


def test_depth_conditioned_spikes_are_locked_and_fair():
    config = json.loads(Path(
        "configs/bilin18_depth_conditioned_semantic_spikes.json"
    ).read_text())
    source = Path(
        "scripts/run_bilin18_depth_conditioned_semantic_spikes.py"
    ).read_text()
    assert "locked" in config["status"]
    assert config["layers"] == [0, 3, 8, 17]
    assert config["random_subspaces"] >= 32
    assert config["null_replicates"] >= 32
    for required in (
        "canonicalize", "random_subspace", "sign_null", "pairing_null",
        "global_top4", "same_family_overlap", "wrong_family_overlap",
        "fit_powerlaw", "empirical_p",
    ):
        assert required in source
