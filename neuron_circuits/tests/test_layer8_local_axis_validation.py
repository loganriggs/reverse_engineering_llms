import json
from pathlib import Path


def test_local_axis_validation_has_scales_and_controls():
    config = json.loads(Path("configs/bilin18_layer8_local_axis_validation.json").read_text())
    assert config["edit_norm_fractions"] == [0.001, 0.01, 0.05]
    assert config["axes"] >= 4
    assert config["random_controls"] >= 4
