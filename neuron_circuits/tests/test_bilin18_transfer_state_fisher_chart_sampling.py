import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_redteam_is_locked_after_both_runs():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_fisher_chart_sampling_redteam.json").read_text())
    assert config["status"] == "locked-after-base-and-highsample-fisher-chart-runs-before-comparison"
    assert (ROOT / config["base"]).exists()
    assert (ROOT / config["highsample"]).exists()
