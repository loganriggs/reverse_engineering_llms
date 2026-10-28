import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_bilin12_layer11_router_seed_stability import (  # noqa: E402
    best_expert_permutation,
    topk_jaccard,
)


def test_best_expert_permutation_recovers_known_relabeling():
    anchor = torch.linalg.qr(torch.randn(30, 4), mode="reduced").Q
    tails = []
    for _ in range(4):
        candidate = torch.randn(30, 5)
        candidate -= anchor @ (anchor.T @ candidate)
        tails.append(torch.linalg.qr(candidate, mode="reduced").Q)
    bases = torch.stack([torch.cat((anchor, tail), 1) for tail in tails])
    relabeling = (2, 0, 3, 1)
    candidate = bases[list(relabeling)]
    permutation, score = best_expert_permutation(bases, candidate, 4)
    assert score > 0.999
    for reference, candidate_index in enumerate(permutation):
        assert torch.linalg.svdvals(
            bases[reference, :, 4:].T @ candidate[candidate_index, :, 4:]
        ).mean() > 0.999


def test_topk_jaccard_handles_exact_and_disjoint_sets():
    first = torch.tensor([[0, 1], [2, 3]])
    assert topk_jaccard(first, first) == 1.0
    second = torch.tensor([[2, 3], [0, 1]])
    assert topk_jaccard(first, second) == 0.0


def test_config_locks_three_seed_alignment_and_behavior_tests():
    config = json.loads(Path("configs/bilin12_layer11_router_seed_stability.json").read_text())
    assert "locked" in config["status"]
    assert len(config["seeds"]) == 3
    assert len(config["hypotheses"]) == 7
