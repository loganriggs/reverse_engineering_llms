import json
from pathlib import Path


def test_independent_spectrum_replication_is_frozen_and_has_nulls():
    config = json.loads(Path("configs/bilin12_checkpoint_spectrum_replication.json").read_text())
    assert config["layers"] == [0, 5, 11]
    assert len(config["snapshot_commit"]) == 40
    assert config["null_replicates"] >= 16
    source = Path("scripts/run_bilin12_checkpoint_spectrum_replication.py").read_text()
    for required in ("symmetric_channel_gram", "sign_nulls", "pairing_nulls", "fit_powerlaw"):
        assert required in source
