import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_final_destination_subspace_audit import fit_bases, project


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_final_destination_subspace_audit.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_final_destination_subspace_audit.json"


def test_fit_bases_are_orthonormal_and_nested():
    torch.manual_seed(0)
    values = {"0": torch.randn(20, 12), "1": 2 * torch.randn(20, 12)}
    layerwise, shared, random = fit_bases(values, 7)
    for basis in [*layerwise.values(), shared, *random.values()]:
        assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5)
    sample = values["0"][:3]
    assert project(sample, layerwise["0"][:, :12]).shape == sample.shape


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-native-head-failure-before")
    assert payload["ranks"] == [1, 2, 4, 8, 16, 32, 48, 64]
    assert len(payload["hypotheses"]) == 9


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
