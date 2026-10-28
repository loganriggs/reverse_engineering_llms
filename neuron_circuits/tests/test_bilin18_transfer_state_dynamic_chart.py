import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_chart_is_locked_after_static_controls():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_dynamic_chart.json").read_text())
    assert config["status"] == "locked-after-static-and-native-basis-controls-before-dynamic-chart-test"
    assert config["ranks"] == [64, 128]
