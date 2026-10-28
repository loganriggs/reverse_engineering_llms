from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer01_new_pair_cluster_redteam import aggregate_pairs, spearman  # noqa: E402


def test_contexts_are_aggregated_before_pair_advantage():
    records = [
        {"pair":"1->2","family":"a","strict":.1,"family_effects":{"a":.4,"b":.2}},
        {"pair":"1->2","family":"a","strict":.2,"family_effects":{"a":.2,"b":.1}},
    ]
    row = aggregate_pairs(records,["a","b"])[0]
    assert abs(row["correct"]-.3) < 1e-9
    assert abs(row["advantage"]-.15) < 1e-9
    assert abs(spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-12
