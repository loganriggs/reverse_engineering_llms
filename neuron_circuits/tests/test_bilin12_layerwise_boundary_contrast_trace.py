import json
from pathlib import Path


def test_layerwise_trace_config_is_locked():
    config = json.loads(Path("configs/bilin12_layerwise_boundary_contrast_trace.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["address_layer"] == 11
    assert len(config["hypotheses"]) == 8


def test_layerwise_trace_has_emergence_and_causal_patches():
    source = Path("scripts/run_bilin12_layerwise_boundary_contrast_trace.py").read_text()
    assert "contrast_energy_ratio" in source
    assert "mixed_attribution" in source
    assert "attention_attribution" in source
    assert "mlp_attribution" in source
    assert "contrast_patch_recovery" in source
    assert "relative_full_token_kl" in source
    assert "run_patched_suffix" in source
