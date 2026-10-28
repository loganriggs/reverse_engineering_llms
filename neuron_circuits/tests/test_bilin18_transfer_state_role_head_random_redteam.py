import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_random_redteam_is_locked_after_single_random_control():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_role_head_random_redteam.json").read_text())
    assert config["status"] == "locked-after-single-random-head-control-before-permutation-redteam"
    assert config["ranks"] == [8, 16]
    assert config["random_draws"] == 16
