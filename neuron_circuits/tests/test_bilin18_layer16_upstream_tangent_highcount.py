import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_upstream_tangent_highcount.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_upstream_tangent_highcount.json"


def test_highcount_has_eight_times_selection_sites():
    config = json.loads(CONFIG.read_text())
    assert config["selection_discovery"]["rows"] == 256
    assert config["budgets"][-1] == 4608


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
