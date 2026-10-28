import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_gender_path_singularity_redteam import path_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_gender_path_singularity_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_gender_path_singularity_redteam.json"


def test_path_endpoints_match_for_straight_and_spherical_paths():
    torch.manual_seed(3)
    left = torch.randn(1, 4, 8)
    right = left.clone()
    right[:, 2] = torch.randn(1, 8)
    right[:, 2] *= left[:, 2].norm() / right[:, 2].norm()
    for spherical in (False, True):
        start, _ = path_state(left, right, 2, 0.0, spherical)
        stop, _ = path_state(left, right, 2, 1.0, spherical)
        torch.testing.assert_close(start, left, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stop, right, atol=1e-5, rtol=1e-5)


def test_preregistration_is_discovery_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("discovery-locked-after-gauss8")
    assert payload["anomalous_pair"] == "465->607"
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
