import json
from pathlib import Path


def test_layer3_frozen_transfer_is_heldout_and_controlled():
    config = json.loads(Path("configs/bilin18_layer3_frozen_channel_transfer.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["rank"] == 256
    assert config["discovery_context_index"] != config["evaluation_context_index"]
    source = Path("scripts/run_bilin18_layer3_frozen_channel_transfer.py").read_text()
    for control in ("oracle", "pair", "family", "global", "wrong_family", "random"):
        assert f'"{control}"' in source
    assert "fisher_fidelity" in source
    assert "support_jaccard" in source
