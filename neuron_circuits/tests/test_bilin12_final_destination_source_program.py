import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bilin12_final_destination_source_program import source_write_contributions


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bilin12_final_destination_source_program.json"
OUTPUT = ROOT / "artifacts/bilin18_tensor_decomposition/bilin12_final_destination_source_program.json"


class Attention:
    lamb = .25
    head_dim = 2
    n_head = 2
    c_proj = torch.nn.Linear(4, 5)


class Block:
    attn = Attention()


def test_source_contributions_sum_to_bias_free_write():
    torch.manual_seed(0)
    batch, length, heads, head_dimension = 3, 4, 2, 2
    pattern = torch.rand(batch, heads, length, length)
    pattern /= pattern.sum(-1, keepdim=True)
    details = {
        "pattern": pattern,
        "current_value": torch.randn(batch, length, heads, head_dimension),
        "value_bus": torch.randn(batch, length, heads, head_dimension),
    }
    contributions = source_write_contributions(Block(), details, 3)
    mixed = .75 * details["current_value"] + .25 * details["value_bus"]
    attended = torch.einsum("bhs,bshd->bhd", pattern[:, :, 3], mixed)
    direct = Block.attn.c_proj(attended.reshape(batch, -1)) - Block.attn.c_proj.bias
    assert torch.allclose(contributions.sum(1), direct, atol=1e-6)


def test_preregistration_is_locked():
    payload = json.loads(CONFIG.read_text())
    assert payload["status"].startswith("locked-after-fixed-subspace-failure-before")
    assert payload["budgets"][-1] == 128
    assert len(payload["hypotheses"]) == 10


def test_completed_output_if_present():
    if not OUTPUT.exists(): return
    payload = json.loads(OUTPUT.read_text())
    assert set(payload["hypotheses"]) == set(json.loads(CONFIG.read_text())["hypotheses"])
    assert all(isinstance(row["supported"], bool) for row in payload["hypotheses"].values())
