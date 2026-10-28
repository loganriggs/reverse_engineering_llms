import json
from pathlib import Path


def test_family_interaction_modes_are_lopo_exact_and_causal():
    config = json.loads(Path("configs/bilin18_layer3_family_interaction_modes.json").read_text())
    assert config["status"].startswith("hypotheses-locked")
    assert config["output_rank"] == 4
    assert 256 in config["interaction_ranks"]
    source = Path("scripts/run_bilin18_layer3_family_interaction_modes.py").read_text()
    assert "candidate_pair != pair" in source
    assert "torch.linalg.eigh" in source
    assert "family_exact" in source
    assert "wrong_family_best" in source
    assert "native_lopo" in source
    assert "fisher_fidelity" in source
