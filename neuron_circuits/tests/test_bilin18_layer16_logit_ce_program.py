import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_logit_ce_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_logit_ce_program.json"


def test_logit_program_declares_behavioral_metrics_and_targets():
    config = json.loads(CONFIG.read_text())
    assert config["final_layer"] == 17
    assert config["inner_ranks"] == [4, 8]
    assert all(spec["rows"] == 8 for spec in config["evaluation"].values())


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
