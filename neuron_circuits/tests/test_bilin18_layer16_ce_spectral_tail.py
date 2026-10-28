import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_ce_spectral_tail.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_spectral_tail.json"


def test_tail_sweep_is_near_full_natural_state_rank():
    config = json.loads(CONFIG.read_text())
    assert config["operator_modes"] == config["budgets"][-1] == 1024
    assert 4 * config["local_atlas_discovery"]["rows"] == 1024


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
