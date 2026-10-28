import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin12_layer01_frozen_taylor_behavior import behavior_metrics, exact_kl, sample_edits


def test_behavior_experiment_is_frozen_and_broad():
    config = json.loads((ROOT / "configs/bilin12_layer01_frozen_taylor_behavior.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["frozen_orders"] == {"0": 3, "1": 5}
    assert config["sites_per_corpus"] >= 64
    assert config["comparison_rank"] in config["ranks"]
    assert "behavior_oracle" in config["methods"]


def test_sample_edits_are_deterministic_and_change_tokens():
    rows = torch.arange(40 * 70).reshape(40, 70).numpy() % 31
    first = sample_edits(rows, 12, 7, 47, 64)
    second = sample_edits(rows, 12, 7, 47, 64)
    assert first == second
    assert len({row for row, _ in first}) == 12
    assert all(rows[row, 47] != target for row, target in first)


def test_behavior_metrics_has_exact_and_base_endpoints():
    base = torch.tensor([0.0, 0.0, 0.0])
    reference = torch.tensor([1.0, -0.5, 0.2])
    methods = {"exact": reference.clone(), "base": base.clone()}
    rows = behavior_metrics(base, reference, reference, methods, label=0)
    assert abs(rows["exact"]["kl_error"]) < 1e-7
    assert abs(rows["exact"]["fisher_fidelity"] - 1.0) < 1e-6
    assert abs(rows["base"]["fisher_fidelity"]) < 1e-6


def test_exact_kl_resolves_tiny_positive_effects():
    reference = torch.linspace(-8, 8, 50304)
    prediction = reference.clone(); prediction[123] += 1e-3
    value = exact_kl(reference, prediction)
    assert value > 0
    assert exact_kl(reference, reference) == 0
