import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_upstream_tangent_channels.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_upstream_tangent_channels.json"


def test_upstream_tangent_test_uses_same_full_channel_budget():
    config = json.loads(CONFIG.read_text())
    assert config["layer"] == 12 and config["layers"] == [16]
    assert config["budgets"][-1] == 4608
    assert Path(ROOT / config["local_pca_artifact"]).exists()


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
