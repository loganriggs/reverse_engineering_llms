import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_directional_order_transfer import choose_order


def test_choose_order_uses_median_not_outlier_mean():
    rows = [
        {"fidelity": {"midpoint_order1": .4, "midpoint_order2": .5}},
        {"fidelity": {"midpoint_order1": .4, "midpoint_order2": .5}},
        {"fidelity": {"midpoint_order1": 2.0, "midpoint_order2": .5}},
    ]
    assert choose_order(rows, [1, 2]) == 2
