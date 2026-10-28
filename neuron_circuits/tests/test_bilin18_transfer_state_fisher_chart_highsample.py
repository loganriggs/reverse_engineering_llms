import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_highsample_fisher_chart_is_locked_after_instability():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_fisher_chart_highsample.json").read_text())
    assert config["status"] == "locked-after-base-fisher-sketch-instability-before-higher-sample-replication"
    assert config["exact_top_tokens"] == 16
    assert config["tail_samples_per_split"] == 16
