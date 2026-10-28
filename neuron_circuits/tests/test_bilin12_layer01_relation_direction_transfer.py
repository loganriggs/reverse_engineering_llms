import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_relation_direction_transfer import donor_sets


def test_donor_sets_exclude_target_and_match_sizes():
    families = {
        "a": [[1, 2], [3, 4], [5, 6]],
        "b": [[7, 8], [9, 10], [11, 12]],
    }
    same, different = donor_sets(families, "a", [1, 2], 4)
    assert (1, 2) not in same
    assert set(same) == {(3, 4), (5, 6)}
    assert len(different) == len(same)
    assert all(pair in {(7, 8), (9, 10), (11, 12)} for pair in different)
