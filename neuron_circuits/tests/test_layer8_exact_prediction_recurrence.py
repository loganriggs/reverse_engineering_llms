import json
from pathlib import Path


def test_exact_prediction_recurrence_is_locked_on_shared_tokens():
    config = json.loads(Path("configs/bilin18_layer8_exact_prediction_recurrence.json").read_text())
    assert config["prediction_token_ids"] == [262, 13, 11, 198]
    assert config["sites_per_token"] >= 8
    assert config["axes"] == 4
