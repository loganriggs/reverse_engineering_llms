import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generic_swap_redteam_is_locked_with_single_token_candidate_panel():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_H2_value_echo_generic_swap_redteam.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert len(config["control_pairs"]) == 16
    assert len({word for pair in config["control_pairs"] for word in pair}) == 32

