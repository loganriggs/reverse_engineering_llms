import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_semantic_layer0_balanced_subspaces import normalized_stack


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_semantic_layer0_balanced_subspaces.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_semantic_layer0_balanced_subspaces.json"


def test_normalized_stack_gives_each_path_equal_energy():
    first = torch.eye(3)
    second = 100 * torch.eye(3)
    stacked = normalized_stack([first, second])
    assert torch.allclose(stacked[:3].square().sum(), torch.tensor(1.0))
    assert torch.allclose(stacked[3:].square().sum(), torch.tensor(1.0))


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-unbalanced-gender-rank8")
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
