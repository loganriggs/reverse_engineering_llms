import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_layer1_key_norm_path_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_layer1_key_norm_path_redteam.json"


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-key-routing-localization")
    assert payload["layer"] == 1
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
