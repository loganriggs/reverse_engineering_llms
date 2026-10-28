import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_mlp_path_highcount_replication.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_mlp_path_highcount_replication.json"


def test_highcount_panels_are_disjoint_and_large():
    config = json.loads(CONFIG.read_text())
    axis = config["axis_discovery"]; discovery = config["path_discovery"]
    evaluation = config["evaluation"]["fineweb"]
    intervals = [(row["start"], row["start"] + row["rows"])
                 for row in (axis, discovery, evaluation)]
    assert all(a[1] <= b[0] or b[1] <= a[0]
               for i, a in enumerate(intervals) for b in intervals[i + 1:])
    assert all(spec["rows"] >= 64 for spec in config["evaluation"].values())


def test_output_matches_locked_config_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(row["sites"] >= 256 for row in payload["results"].values())
