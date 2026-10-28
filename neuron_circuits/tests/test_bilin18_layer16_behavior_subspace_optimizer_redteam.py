import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_layer16_behavior_subspace_optimizer_redteam import (  # noqa: E402
    grassmann_gradient,
    relative_error_objective,
    retract,
)


CONFIG = ROOT / "configs/bilin18_layer16_behavior_subspace_optimizer_redteam.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_behavior_subspace_optimizer_redteam.json"
BASIS = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_layer16_behavior_subspace_optimizer_redteam.pt"


def test_optimizer_redteam_uses_disjoint_panels_and_locked_rates():
    config = json.loads(CONFIG.read_text())
    train = config["subspace_train"]
    validation = config["subspace_validation"]
    assert train["start"] + train["rows"] <= validation["start"]
    assert validation["start"] + validation["rows"] <= config["evaluation"]["fineweb"]["start"]
    assert config["subspace_rank"] == 64
    assert config["learning_rates"] == [0.0001, 0.0003, 0.001, 0.003]


def test_grassmann_step_is_orthonormal_and_descends_locally():
    generator = torch.Generator().manual_seed(17)
    forward = torch.randn(24, 5, 9, generator=generator, dtype=torch.float64)
    effective = torch.randn(24, 9, generator=generator, dtype=torch.float64)
    target = torch.einsum("bnd,bd->bn", forward, effective)
    basis = torch.linalg.qr(torch.randn(9, 3, generator=generator, dtype=torch.float64), mode="reduced").Q
    parameter = basis.clone().requires_grad_(True)
    loss = relative_error_objective(forward, effective, target, parameter)
    loss.backward()
    tangent = grassmann_gradient(parameter.detach(), parameter.grad.detach())
    updated = retract(parameter.detach() - 1e-4 * tangent)
    updated_loss = relative_error_objective(forward, effective, target, updated)
    assert torch.allclose(updated.T @ updated, torch.eye(3, dtype=torch.float64), atol=1e-10)
    assert updated_loss < loss.detach()


def test_output_matches_locked_hypotheses_if_present():
    if not OUTPUT.exists():
        return
    config = json.loads(CONFIG.read_text())
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(config["hypotheses"])
    assert BASIS.exists()
