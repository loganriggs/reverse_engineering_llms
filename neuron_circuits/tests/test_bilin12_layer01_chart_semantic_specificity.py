import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer01_chart_semantic_specificity import summarize_records


def test_summary_preserves_signed_router_gain_and_choice_fraction():
    rows = [
        {
            "semantic_native": .5, "semantic_routed": .7, "semantic_oracle": .8,
            "semantic_gain": .2, "semantic_router_gap": .1, "semantic_native_choice": 0.,
            "synthetic_native": .6, "synthetic_routed": .5, "synthetic_oracle": .9,
            "synthetic_gain": -.1, "synthetic_router_gap": .4, "synthetic_native_choice": .5,
        },
        {
            "semantic_native": .7, "semantic_routed": .7, "semantic_oracle": .9,
            "semantic_gain": 0., "semantic_router_gap": .2, "semantic_native_choice": 1.,
            "synthetic_native": .4, "synthetic_routed": .8, "synthetic_oracle": .8,
            "synthetic_gain": .4, "synthetic_router_gap": 0., "synthetic_native_choice": 0.,
        },
    ]
    summary = summarize_records(rows)
    assert summary["semantic_gain"] == .1
    assert summary["synthetic_gain"] == pytest.approx(.15)
    assert summary["semantic_native_choice"] == .5
    assert summary["synthetic_native_choice"] == .25
    assert summary["synthetic_router_gap"] == pytest.approx(.2)
