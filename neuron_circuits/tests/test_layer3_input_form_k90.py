import json
from pathlib import Path


def test_input_form_k90_is_family_resolved_and_reaches_large_ranks():
    config = json.loads(Path("configs/bilin18_layer3_input_form_k90.json").read_text())
    assert config["ranks"][-1] == 512
    assert config["scales"] == [0.1, 1.0]
    source = Path("scripts/run_bilin18_layer3_input_form_k90.py").read_text()
    assert "by_family" in source
    assert "causal_k90" in source
    assert "input_metric" in source
    assert "evaluate_writes" in source
