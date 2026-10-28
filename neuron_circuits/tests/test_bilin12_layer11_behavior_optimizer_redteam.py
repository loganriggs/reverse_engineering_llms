import json
from pathlib import Path


def test_optimizer_redteam_locks_local_sweep_and_external_evaluation():
    config = json.loads(Path("configs/bilin12_layer11_behavior_optimizer_redteam.json").read_text())
    assert "locked" in config["status"]
    assert config["learning_rates"] == [0.003, 0.0003, 0.00003]
    assert config["initialization_noise"] == 0.0
    assert config["optimization_steps"] >= 500
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert len(config["hypotheses"]) == 5


def test_optimizer_redteam_selects_only_on_validation():
    source = Path("scripts/run_bilin12_layer11_behavior_optimizer_redteam.py").read_text()
    assert 'min(rows, key=lambda item: item["best_validation_kl"])' in source
    assert "behavior_metrics" in source
    assert "variance_fraction" in source
