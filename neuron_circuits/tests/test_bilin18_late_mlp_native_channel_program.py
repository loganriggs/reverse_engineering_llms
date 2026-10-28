import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_late_mlp_native_channel_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_late_mlp_native_channel_program.json"


def test_native_channel_budget_covers_complete_bank():
    config = json.loads(CONFIG.read_text())
    assert config["budgets"][-1] == 4608
    assert config["oracle_budget"] in config["budgets"]
    assert config["layers"] == [13, 14, 15, 16]


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert set(payload["ranked_channel_prefixes"]) == {"13", "14", "15", "16"}
