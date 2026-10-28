import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/bilin12_semantic_source_individual_redteam.json"
OUTPUT=ROOT/"artifacts/bilin18_tensor_decomposition/bilin12_semantic_source_individual_redteam.json"

def test_preregistration_is_locked_and_fresh():
    payload=json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-aggregate-semantic-transfer-before-fresh")
    assert payload["skip_contexts_per_pair"]==2
    assert len(payload["hypotheses"])==10

def test_completed_output_if_present():
    if not OUTPUT.exists():return
    payload=json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"])==set(json.loads(CONFIG.read_text())["hypotheses"])
