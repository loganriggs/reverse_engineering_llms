import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/bilin12_newline_source_scale_redteam.json"
OUTPUT=ROOT/"artifacts/bilin18_tensor_decomposition/bilin12_newline_source_scale_redteam.json"

def test_preregistration_is_locked():
    payload=json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-semantic-individual-result-before")
    assert payload["scales"]==[0.125,0.25,0.5,1.0,1.5]
    assert len(payload["hypotheses"])==10

def test_completed_output_if_present():
    if not OUTPUT.exists():return
    payload=json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"])==set(json.loads(CONFIG.read_text())["hypotheses"])
