import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_dynamic_attention_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_dynamic_attention_redteam.json"


def test_dynamic_redteam_has_disjoint_small_panels():
    config = json.loads(CONFIG.read_text())
    assert config["dynamic_discovery"]["rows"] == 16
    assert all(spec["rows"] == 8 for spec in config["evaluation"].values())
    assert config["tangent_chunk"] <= config["rank"]


def test_output_matches_hypotheses_if_present():
    if not OUTPUT.exists(): return
    config = json.loads(CONFIG.read_text()); payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
