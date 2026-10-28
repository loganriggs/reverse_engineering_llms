import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_attention_factor_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_attention_factor_program.json"


def test_factor_inventory_has_identity_and_two_roles_per_layer():
    parent = json.loads((ROOT / json.loads(CONFIG.read_text())["source_config"]).read_text())
    base = json.loads((ROOT / parent["source_config"]).read_text())
    layers = base["final_layer"] - base["layer"]
    assert 1 + 2 * layers == 11


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert len(payload["factor_order"]) == 11
