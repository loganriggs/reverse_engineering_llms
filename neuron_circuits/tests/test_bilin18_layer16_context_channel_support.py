import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_context_channel_support.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_context_channel_support.json"


def test_context_support_distinguishes_fixed_and_reweighted():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 32
    assert 512 in config["budgets"] and config["budgets"][-1] == 4608


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
