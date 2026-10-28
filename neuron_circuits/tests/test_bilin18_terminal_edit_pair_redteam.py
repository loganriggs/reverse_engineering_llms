import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pair_redteam_protocol_is_locked_and_complete():
    config = json.loads(
        (ROOT / "configs/bilin18_terminal_edit_pair_redteam.json").read_text()
    )
    parent = json.loads((ROOT / config["parent_config"]).read_text())
    assert config["status"].startswith("locked-")
    assert len(parent["control_pairs"]) == 16
    assert config["minimum_pairs_for_three_quarters_gate"] == 12
    assert config["minimum_examples_per_cell"] >= 20
