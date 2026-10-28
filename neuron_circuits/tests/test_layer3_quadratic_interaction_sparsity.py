import json
from pathlib import Path


def test_quadratic_interaction_sparsity_has_diag_offdiag_and_rank_controls():
    config = json.loads(Path("configs/bilin18_layer3_quadratic_interaction_sparsity.json").read_text())
    assert config["budgets"][-1] >= 32768
    source = Path("scripts/run_bilin18_layer3_quadratic_interaction_sparsity.py").read_text()
    assert "torch.triu_indices" in source
    assert "offdiagonal" in source
    assert "activation_order" in source
    assert "weight_order" in source
    assert "random_order" in source
    assert "causal_k90" in source
