import json
from pathlib import Path


def test_joint_redteam_is_locked_after_positive_result_and_uses_new_split():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/bilin18_typed_attention_edge_joint_composition_redteam.json").read_text())
    assert config["status"] == "locked-after-positive-joint-composition-before-random-and-all-type-controls"
    assert config["evaluation_start"] == 17000
    assert {"random128", "all"}.issubset(config["methods"])
