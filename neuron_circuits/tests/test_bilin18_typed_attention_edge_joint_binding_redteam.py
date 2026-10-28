import json
from pathlib import Path


def test_joint_binding_redteam_uses_third_split_and_all_coordinates():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/bilin18_typed_attention_edge_joint_binding_redteam.json").read_text())
    assert config["evaluation_start"] == 18000
    assert config["status"].startswith("locked-after-random")
    assert all(f"{axis}_2" in config["methods"] for axis in ("category", "distance", "head"))
    assert "layer_shift" in config["methods"]
