import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_late_mlp_operator_atlas.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_late_mlp_operator_atlas.json"


def test_atlas_uses_disjoint_fineweb_ranges_and_declared_layers():
    config = json.loads(CONFIG.read_text())
    assert config["layers"] == [13, 14, 15, 16]
    ranges = []
    for key in ("axis_discovery", "atlas_discovery", "ridge_validation"):
        row = config[key]
        ranges.append((row["start"], row["start"] + row["rows"]))
    row = config["evaluation"]["fineweb"]
    ranges.append((row["start"], row["start"] + row["rows"]))
    assert all(a1 <= b0 or b1 <= a0 for i, (a0, a1) in enumerate(ranges)
               for b0, b1 in ranges[i + 1:])
    assert max(config["spectral_budgets"]) <= 4 * config["atlas_discovery"]["rows"]


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert set(payload["results"]) == {"13", "14", "15", "16"}
