import json
from pathlib import Path


def test_current_value_recurrence_uses_disjoint_reference_and_query_contexts():
    config = json.loads(
        Path("configs/bilin18_layer4_current_value_code_recurrence.json").read_text()
    )
    assert config["layer"] == 4
    assert config["reference_context_index"] != config["query_context_index"]
    assert Path(config["frozen_results"]).exists()
    source = Path("scripts/run_bilin18_layer4_current_value_code_recurrence.py").read_text()
    assert "source_current_value" in source
    assert "routed_current_value" in source
    assert "pair_accuracy" in source
    assert "family_gap" in source
