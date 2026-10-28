import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_layer0_mlp_path_subspace import projection_fidelity, subspace_alignment, svd_summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_layer0_mlp_path_subspace.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_layer0_mlp_path_subspace.json"


def test_svd_summary_and_projection_on_rank_two_matrix():
    torch.manual_seed(4)
    matrix = torch.randn(20, 2) @ torch.randn(2, 7)
    summary, basis = svd_summary(matrix, [1, 2, 4])
    assert summary["rank90"] <= 2
    assert projection_fidelity(matrix, basis[:2]) > 0.99999
    assert subspace_alignment(basis[:2], basis[:2]) > 0.99999


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-native-channel-diffuseness")
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
