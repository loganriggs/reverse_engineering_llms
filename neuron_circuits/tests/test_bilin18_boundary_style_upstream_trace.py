import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_upstream_trace import residual_components  # noqa: E402


def test_upstream_trace_is_locked_before_results():
    config = json.loads((ROOT / "configs/bilin18_boundary_style_upstream_trace.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["quadrature_points"] >= 32
    assert len(config["hypotheses"]) == 7


def test_residual_telescope_scaling_on_scalar_fake_model():
    class Block:
        def __init__(self, carry, inject):
            self.lambdas = torch.tensor([carry, inject])
    class Model:
        blocks = [Block(.8 + .01 * i, .2) for i in range(10)]
    dx0 = torch.tensor([[0.0, 0.0]])
    attention = [torch.tensor([[i + 1.0, 0.0]]) for i in range(9)]
    mlp = [torch.tensor([[0.0, i + 2.0]]) for i in range(9)]
    pieces = residual_components(Model(), dx0, attention, mlp, 9)
    x = dx0
    for layer in range(9):
        x = Model.blocks[layer].lambdas[0] * x + Model.blocks[layer].lambdas[1] * dx0 + attention[layer] + mlp[layer]
    target = Model.blocks[9].lambdas[0] * x + Model.blocks[9].lambdas[1] * dx0
    torch.testing.assert_close(sum(pieces.values()), target)


def test_trace_uses_paired_batch_source_indexing():
    source = (ROOT / "scripts/run_bilin18_boundary_style_upstream_trace.py").read_text()
    assert 'mixed[paired, source]' in source
    assert 'details["write"][paired + batch_size, source]' in source
    assert 'details["score1"][paired, head, destination, source]' in source
    assert 'mixed[:batch_size, source]' not in source


def test_completed_trace_recomposes_without_embedding_leakage():
    result = json.loads((ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_upstream_trace.json").read_text())
    assert len(result["results"]) == 6
    assert result["hypotheses"]["U0_telescope_and_value_decomposition_recompose"]["supported"]
    assert all(row["embedding_energy_ratio"] == 0 for row in result["results"])
