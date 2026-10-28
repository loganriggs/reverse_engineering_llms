import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_conditional_output_router import (  # noqa: E402
    ConditionalOutputRouter,
    route_summary,
)


def test_expert_bases_preserve_anchor_and_are_orthonormal():
    anchor = torch.linalg.qr(torch.randn(19, 3), mode="reduced").Q
    initial = torch.randn(4, 19, 5)
    model = ConditionalOutputRouter(anchor, initial, router_rank=7, seed=4)
    bases = model.expert_bases()
    assert bases.shape == (4, 19, 8)
    for basis in bases:
        assert torch.allclose(basis[:, :3], anchor, atol=1e-6)
        assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-5)


def test_route_summary_distinguishes_balanced_and_collapsed_routes():
    balanced = torch.eye(4).repeat(5, 1)
    collapsed = torch.tensor([[1.0, 0, 0, 0]]).repeat(20, 1)
    assert route_summary(balanced)["effective_experts"] > 3.99
    assert route_summary(collapsed)["effective_experts"] < 1.01


def test_config_locks_union_matched_conditional_test_and_ood_controls():
    config = json.loads(Path("configs/bilin12_layer11_conditional_output_router.json").read_text())
    assert "locked" in config["status"]
    assert config["experts"] * config["tail_rank_per_expert"] == 64
    assert config["optimization_steps"] >= 800
    assert len(config["positions"]) == 16
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert len(config["hypotheses"]) == 9


def test_script_has_hard_soft_oracle_and_route_specificity_controls():
    source = Path("scripts/run_bilin12_layer11_conditional_output_router.py").read_text()
    for required in (
        "hard_st", "oracle", "shuffled_hard", "position_hard",
        "fixed_tail16", "fixed_tail64", 'f"expert_{i}"',
    ):
        assert required in source
