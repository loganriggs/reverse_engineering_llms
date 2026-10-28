import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_edit_mode_origin_is_locked_before_results():
    config = json.loads(
        (ROOT / "configs/bilin18_boundary_style_H2_edit_mode_origin.json").read_text()
    )
    assert config["status"].startswith("locked-")
    assert config["head"] == 2

