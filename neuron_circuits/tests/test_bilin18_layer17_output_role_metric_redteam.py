import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_output_role_metric_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_output_role_metric_redteam.json"


def test_metric_redteam_is_locked_to_semantic_atlas():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 16
    assert Path(ROOT / config["semantic_result_path"]).exists()
    assert config["discovery"] == json.loads((ROOT / "configs/bilin18_layer17_tangent_output_semantics.json").read_text())["discovery"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
