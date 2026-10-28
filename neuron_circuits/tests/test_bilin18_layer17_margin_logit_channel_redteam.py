import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_margin_logit_channel_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_margin_logit_channel_redteam.json"


def test_margin_logit_redteam_is_disjoint_and_locked():
    config = json.loads(CONFIG.read_text())
    discovery = config["selection_discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["budgets_per_logit"] == [16, 32, 64, 128, 256, 512]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
