import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bilin18_transfer_state_signed_role_address import (  # noqa: E402
    demand_similarity,
    role_specs,
)


def test_signed_role_address_is_locked_after_qk_failure():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_signed_role_address.json").read_text())
    assert config["status"] == "locked-after-qk-rollout-failure-before-signed-role-address-test"
    assert config["site_count"] == 8


def test_highsample_replication_is_locked_after_base_result():
    config = json.loads((ROOT / "configs/bilin18_transfer_state_signed_role_address_highsample.json").read_text())
    assert config["status"] == "locked-after-base-signed-role-result-before-highsample-replication"
    assert config["exact_top_tokens"] == 16
    assert config["tail_samples_per_split"] == 16
    assert config["seed"] != 20261321


def test_role_specs_separate_mlp_and_attention_families():
    specs = role_specs()
    assert specs["carry"] == {"mlp": False, "roles": set()}
    assert specs["mlp"] == {"mlp": True, "roles": set()}
    assert specs["queries"]["roles"] == {"q1", "q2"}
    assert specs["keys"]["roles"] == {"k1", "k2"}
    assert specs["value"]["roles"] == {"value"}
    assert len(specs["full"]["roles"]) == 5


def test_demand_similarity_recovers_identical_sites():
    adjoint = torch.zeros(2, 1, 4, 3)
    adjoint[:, :, 1, 0] = 2
    adjoint[:, :, 3, 0] = 1
    weights = torch.ones(2)
    result = demand_similarity(adjoint, adjoint, weights, 2)
    assert result["top_jaccard"] == 1
    assert result["cosine"] == 1
    assert set(result["sites"]) == {1, 3}
