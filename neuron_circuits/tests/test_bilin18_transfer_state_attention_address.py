import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_attention_address import (  # noqa: E402
    attention_address_scores,
    top_jaccard,
)


def test_attention_address_is_locked_after_dynamic_selector():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_attention_address.json").read_text())
    assert config["status"] == "locked-after-dynamic-site-selection-before-attention-address-test"
    assert config["site_count"] == 8


def test_rollout_propagates_query_demand_to_source():
    pattern = torch.eye(4)
    pattern[3] = torch.tensor([1., 0., 0., 0.])
    scores = attention_address_scores([pattern, pattern], 0, [3])
    assert scores["direct"].argmax().item() == 0
    assert scores["rollout"].argmax().item() in (0, 3)


def test_top_jaccard():
    assert top_jaccard([1, 2, 3], [2, 3, 4]) == .5
