import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_logit_ce_highcount.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_logit_ce_highcount.json"


def test_highcount_panel_is_four_times_pilot():
    config = json.loads(CONFIG.read_text())
    assert all(spec["rows"] == 32 for spec in config["evaluation"].values())


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
