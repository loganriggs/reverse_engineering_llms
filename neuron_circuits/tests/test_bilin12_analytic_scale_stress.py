import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_router_analytic_scale_stress.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_router_analytic_scale_stress.json"


def energy_fidelity(target, prediction):
    return 1 - (target - prediction).square().sum() / target.square().sum()


def test_one_sided_metric_exposes_even_order_error_hidden_by_central_difference():
    # f(d) = d + d^2.  A Jacobian predicts only +/-d.  The central difference
    # cancels the quadratic term exactly, while separate finite edits retain it.
    delta = torch.tensor([[0.5]])
    plus = delta + delta.square()
    minus = -delta + delta.square()
    tangent = delta

    central_target = (plus - minus) / 2
    assert energy_fidelity(central_target, tangent) == 1

    one_sided_target = torch.cat([plus, minus])
    one_sided_prediction = torch.cat([tangent, -tangent])
    assert energy_fidelity(one_sided_target, one_sided_prediction) < 0.9


def test_scale_stress_declares_the_metric_correction():
    payload = json.loads(CONFIG.read_text())
    assert "one-sided" in payload["status"]
    assert all("one-sided" in text.lower() for text in payload["hypotheses"].values())


def test_completed_scale_stress_matches_config_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    output = json.loads(OUTPUT.read_text())
    assert set(output["hypotheses"]) == set(config["hypotheses"])
    assert all(
        "central_difference_diagnostic" in scale_row
        for corpus_row in output["results"].values()
        for scale_row in corpus_row.values()
    )
