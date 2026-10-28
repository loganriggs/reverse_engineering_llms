import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_his_her_layer0_mlp_path import minimum_budget, squared_fidelity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_his_her_layer0_mlp_path.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_his_her_layer0_mlp_path.json"


def test_channel_budget_finds_exact_orthogonal_reconstruction():
    contributions = torch.eye(4)[None]
    target = contributions.sum(1)
    order = torch.arange(4)
    assert minimum_budget(target, contributions, order, 0.74) == 3
    assert squared_fidelity(target, contributions[:, :3].sum(1)) == 0.75


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-qk-normalization-falsified")
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
