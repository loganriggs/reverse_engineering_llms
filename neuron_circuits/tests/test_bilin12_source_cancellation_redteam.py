import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_source_cancellation_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_source_cancellation_redteam.json"


def test_preregistration_is_locked_on_fresh_rows():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-source-program-before-fresh")
    assert "balanced_fisher" in payload["methods"]
    assert payload["evaluation"]["fineweb"]["start"] == 19656
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
