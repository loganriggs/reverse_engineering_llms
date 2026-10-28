import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_ce_native_channels.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_native_channels.json"


def test_native_loss_test_includes_complete_channel_bank():
    config = json.loads(CONFIG.read_text())
    assert config["budgets"][-1] == 4608
    assert config["oracle_budget"] == 512


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
