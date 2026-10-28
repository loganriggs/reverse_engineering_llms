import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_his_her_layer1_qkov_path import project_heads


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_his_her_layer1_qkov_path.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_his_her_layer1_qkov_path.json"


class _Projection:
    def __init__(self, weight):
        self.weight = weight


class _Attention:
    def __init__(self, weight, heads, head_dimension):
        self.c_proj = _Projection(weight)
        self.n_head = heads
        self.head_dim = head_dimension


class _Block:
    def __init__(self, weight, heads, head_dimension):
        self.attn = _Attention(weight, heads, head_dimension)


def test_project_heads_preserves_head_sum():
    torch.manual_seed(11)
    batch, heads, sources, head_dimension = 2, 3, 5, 4
    dimension = heads * head_dimension
    values = torch.randn(batch, heads, sources, head_dimension)
    weight = torch.randn(dimension, dimension)
    actual = project_heads(_Block(weight, heads, head_dimension), values).sum((1, 2))
    expected = values.sum(2).reshape(batch, dimension) @ weight.T
    torch.testing.assert_close(actual, expected)


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-layer1-routing-localization")
    assert payload["layer"] == 1
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
