import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_semantic_source_jvp_predictor import cosine, projected_source_writes


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_semantic_source_jvp_predictor.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_semantic_source_jvp_predictor.json"


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


def test_projected_source_writes_sum_to_attention_write():
    generator = torch.Generator().manual_seed(17)
    batch, heads, length, head_dimension = 2, 3, 5, 4
    dimension = heads * head_dimension
    scores = torch.rand(batch, heads, length, length, generator=generator)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    pattern = scores.masked_fill(~causal, 0)
    pattern = pattern / pattern.sum(-1, keepdim=True)
    values = torch.randn(batch, length, heads, head_dimension, generator=generator)
    weight = torch.randn(dimension, dimension, generator=generator)
    block = _Block(weight, heads, head_dimension)
    destination = length - 1

    sources = projected_source_writes(block, pattern, values, destination)
    attended = torch.einsum("bhk,bkhd->bhd", pattern[:, :, destination], values)
    expected = attended.reshape(batch, dimension) @ weight.T
    torch.testing.assert_close(sources.sum(1), expected)


def test_cosine_flattens_source_and_residual_axes():
    left = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert torch.allclose(cosine(left, left), torch.ones(1))
    assert torch.allclose(cosine(left, -left), -torch.ones(1))


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"] == "locked-before-semantic-source-jvp-run"
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists():
        return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
