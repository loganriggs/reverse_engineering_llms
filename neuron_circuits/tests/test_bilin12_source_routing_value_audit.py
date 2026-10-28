import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_source_routing_value_audit import product_shapley_parts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_source_routing_value_audit.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_source_routing_value_audit.json"


def test_product_shapley_parts_recompose_product_change():
    torch.manual_seed(0)
    a0 = torch.randn(2, 3, 4); a1 = torch.randn(2, 3, 4)
    v0 = torch.randn(2, 3, 4, 5); v1 = torch.randn(2, 3, 4, 5)
    routing, value = product_shapley_parts(a0, a1, v0, v1)
    target = a1[..., None] * v1 - a0[..., None] * v0
    assert torch.allclose(routing + value, target, atol=1e-6)


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-cooperative-gram-result-before")
    assert payload["source_budget"] == 32
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
