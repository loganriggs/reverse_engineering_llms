import json
from pathlib import Path


def test_upstream_config_is_locked():
    config = json.loads(Path("configs/bilin12_layer11_boundary_upstream_decomposition.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["layer"] == 11
    assert len(config["hypotheses"]) == 8


def test_upstream_script_uses_factorial_vector_shapley():
    source = Path("scripts/run_bilin12_layer11_boundary_upstream_decomposition.py").read_text()
    assert "carried_residual" in source
    assert "direct_embedding" in source
    assert "attention" in source
    assert "vector_shapley" in source
    assert 'accumulate_metric(store,"contrast"' in source
    assert 'accumulate_metric(store,"fisher_logit"' in source
    assert "additive_residual" in source
