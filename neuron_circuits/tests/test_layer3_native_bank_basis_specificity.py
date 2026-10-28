import json
from pathlib import Path


def test_native_bank_basis_specificity_has_global_and_all_wrong_controls():
    config = json.loads(Path("configs/bilin18_layer3_native_bank_basis_specificity.json").read_text())
    assert config["ranks"] == [4, 16, 32, 64]
    source = Path("scripts/run_bilin18_layer3_native_bank_basis_specificity.py").read_text()
    assert "global_basis" in source
    assert "wrong_best" in source
    assert "canonical_cosine" in source
    assert "candidate_pair != pair" in source
    assert "evaluate_writes" in source
