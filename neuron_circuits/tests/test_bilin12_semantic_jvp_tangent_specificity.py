import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_semantic_jvp_tangent_specificity.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_semantic_jvp_tangent_specificity.json"


def test_preregistration_is_locked_after_jvp_result():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"] == "locked-after-jvp-support-result-before-tangent-specificity-redteam"
    assert payload["random_probes"] == 4
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
