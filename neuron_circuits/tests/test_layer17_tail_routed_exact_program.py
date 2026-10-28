import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_bilin18_layer17_tail_routed_exact_program import RoutedTensorProgram  # noqa: E402


def test_routed_program_soft_and_hard_shapes_and_gates():
    generator = torch.Generator().manual_seed(9)
    base = torch.linalg.qr(torch.randn(12, 3, generator=generator)).Q
    model = RoutedTensorProgram(12, 3, 4, 5, base, 11)
    coordinates = torch.randn(7, 12, generator=generator)
    forms = torch.randn(2, 12, 12, generator=generator)
    forms = 0.5 * (forms + forms.transpose(1, 2))
    soft, soft_gates = model(coordinates, forms, hard=False)
    hard, hard_gates = model(coordinates, forms, hard=True)
    assert soft.shape == hard.shape == (7, 2)
    assert torch.allclose(soft_gates.sum(1), torch.ones(7), atol=1e-6)
    assert torch.allclose(hard_gates.sum(1), torch.ones(7), atol=1e-6)
    assert torch.all((hard_gates == 0) | (hard_gates == 1))


def test_routed_exact_program_is_locked_with_required_controls():
    config = json.loads(Path(
        "configs/bilin18_layer17_tail_routed_exact_program.json"
    ).read_text())
    source = Path("scripts/run_bilin18_layer17_tail_routed_exact_program.py").read_text()
    assert "locked" in config["status"]
    assert config["experts"] >= 4 and config["expert_rank"] == 8
    assert config["exact_steps"] >= 100
    for required in (
        "global32", "fisher_soft", "kl_soft", "kl_hard", "ce_soft",
        "exact_kl", "ce_delta", "effective_experts", "opposite_half_fisher",
    ):
        assert required in source
