import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_ce_native_greedy_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_native_greedy_redteam.json"


def test_greedy_redteam_uses_complete_bank_and_retained_baseline():
    config = json.loads(CONFIG.read_text())
    assert config["budgets"][-1] == 4608
    assert (ROOT / config["marginal_artifact"]).exists()


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert len(payload["greedy_channel_order"]) == 4608
