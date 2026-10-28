import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_rotation_selection_transfer import best_seed, pair_means


def test_rotation_selection_excludes_heldout_pair():
    rows = [
        {"pair": "a", "rotations": {"1": 0.0, "2": 10.0}},
        {"pair": "b", "rotations": {"1": 4.0, "2": 1.0}},
        {"pair": "c", "rotations": {"1": 5.0, "2": 1.0}},
    ]
    assert best_seed(rows, [1, 2], excluded_pair="a") == 1
    means = pair_means(rows, lambda row: row["rotations"]["1"])
    assert means == {"a": 0.0, "b": 4.0, "c": 5.0}
