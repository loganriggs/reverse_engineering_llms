import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_router_topk_frontier import combination_bases  # noqa: E402


def test_combination_bases_have_expected_count_rank_and_orthogonality():
    anchor = torch.linalg.qr(torch.randn(23, 4), mode="reduced").Q
    tails = []
    for _ in range(4):
        candidate = torch.randn(23, 3)
        candidate -= anchor @ (anchor.T @ candidate)
        tails.append(torch.linalg.qr(candidate, mode="reduced").Q)
    expert_bases = torch.stack([torch.cat((anchor, tail), 1) for tail in tails])
    subsets, combined = combination_bases(expert_bases, anchor_rank=4, k=2)
    assert len(subsets) == 6
    assert combined.shape == (6, 23, 10)
    for basis in combined:
        assert torch.allclose(basis.T @ basis, torch.eye(10), atol=1e-5)


def test_config_locks_complete_frozen_topk_frontier():
    config = json.loads(Path("configs/bilin12_layer11_router_topk_frontier.json").read_text())
    assert "locked" in config["status"]
    assert config["topk"] == [1, 2, 3, 4]
    assert config["experts"] == 4
    assert set(config["evaluation"]) == {"fineweb", "wikitext", "tinystories"}
    assert len(config["hypotheses"]) == 8


def test_script_has_router_oracle_shuffle_and_fixed_rank_controls():
    source = Path("scripts/run_bilin12_layer11_router_topk_frontier.py").read_text()
    for required in ("router_top", "oracle_top", "shuffled_top", "fixed_rank"):
        assert required in source
