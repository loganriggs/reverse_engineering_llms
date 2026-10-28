import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer17_context_source_mediation_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer17_context_source_mediation_redteam.json"


def test_source_mediation_redteam_is_locked_and_disjoint():
    config = json.loads(CONFIG.read_text())
    assert config["rank"] == 16
    assert len(config["positions"]) == 16
    fineweb = config["evaluation"]["fineweb"]
    discovery = config["discovery"]
    assert fineweb["start"] + fineweb["rows"] <= discovery["start"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
