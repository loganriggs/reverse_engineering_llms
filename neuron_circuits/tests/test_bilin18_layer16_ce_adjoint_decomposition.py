import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin18_layer16_ce_adjoint_decomposition.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_adjoint_decomposition.json"
FACTORS = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_ce_adjoint_factors.pt"


def test_ce_adjoint_decomposition_uses_disjoint_panels():
    config = json.loads(CONFIG.read_text())
    discovery = config["decomposition_discovery"]
    fineweb = config["evaluation"]["fineweb"]
    assert discovery["start"] + discovery["rows"] <= fineweb["start"]
    assert config["subspace_ranks"] == [32, 64, 128, 256]
    assert set(config["hypotheses"]) == {
        f"D{index}_{suffix}" for index, suffix in enumerate((
            "adjoint_decomposition_is_exact",
            "target_readout_dominates_ce_energy",
            "prediction_term_has_compact_forward_interface",
            "target_term_is_broad_at_rank64",
            "model_margin_has_compact_forward_interface",
            "target_adjoint_is_broader_than_prediction_adjoint",
            "target_identity_predicts_repeated_target_reads",
            "target_identity_clusters_effective_adjoints",
            "ce_breadth_is_primarily_target_readout_breadth",
            "target_prototypes_transfer_across_contexts",
        ))
    }


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert FACTORS.exists()
