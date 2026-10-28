import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_chart_rank_frontier import subset_rows


def test_subset_enumeration_counts_are_exact():
    rows = subset_rows(["a", "b", "c", "d"], [0, 1, 2, 4])
    assert [len(rows[count]) for count in (0, 1, 2, 4)] == [1, 4, 6, 1]
