import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_position_role_redteam_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layer1_cross_position_role_redteam.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["prediction_position"] == 127
    assert config["edit_positions"] == [63, 95, 119]
    assert all(position < config["prediction_position"] for position in config["edit_positions"])

