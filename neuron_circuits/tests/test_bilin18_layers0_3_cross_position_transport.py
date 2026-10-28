import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_position_depth_redteam_is_locked():
    config = json.loads(
        (ROOT / "configs/bilin18_layers0_3_cross_position_transport.json").read_text()
    )
    assert config["status"] == "locked-after-layers0-3-same-position-replication-before-cross-position-redteam"
    assert config["source_layers"] == [0, 1, 2, 3]
    assert config["edit_positions"] == [63, 95, 119]
    assert config["prediction_position"] == 127
    assert config["support_rank"] == 64
