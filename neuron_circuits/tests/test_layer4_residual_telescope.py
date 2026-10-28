import json
from pathlib import Path


def test_residual_telescope_is_exact_and_uses_analytic_integration():
    config = json.loads(Path("configs/bilin18_layer4_residual_telescope.json").read_text())
    assert config["layer"] == 4
    assert config["quadrature_points"] >= 8
    assert Path(config["frozen_results"]).exists()
    source = Path("scripts/run_bilin18_layer4_residual_telescope.py").read_text()
    assert "initial_embedding" in source
    assert "repeat_embedding_l4" in source
    assert "attention_l0" in source
    assert "mlp_l3" in source
    assert "leggauss" in source
    assert "rmsnorm_jvp" in source
