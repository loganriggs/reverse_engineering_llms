import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_source_fisher_gram_program import cooperative_order


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_source_fisher_gram_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_source_fisher_gram_program.json"


def test_cooperative_order_exactly_recovers_orthogonal_target():
    probability = torch.tensor([[.5, .5]])
    effects = torch.tensor([[[1., -1.], [-1., 1.]]])
    target = effects.sum(1)
    order, frontier, signs = cooperative_order(probability, target, effects)
    assert sorted(order[0].tolist()) == [0, 1]
    assert frontier.shape == (1, 2)
    assert signs.shape == (1, 2)


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-cancellation-replication-before")
    assert payload["evaluation"]["fineweb"]["start"] == 19672
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
