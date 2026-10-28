import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bilin18_boundary_style_layer8_builder import RELATIONS, relation_masks  # noqa: E402


def test_layer8_builder_is_locked_before_results():
    config = json.loads((ROOT / "configs/bilin18_boundary_style_layer8_builder.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["quadrature_points"] >= 32
    assert len(config["hypotheses"]) == 7


def test_source_relations_partition_causal_prefix():
    masks = relation_masks(8, torch.tensor([2, 3]), torch.tensor([6, 7]))
    total = sum(mask.to(torch.int64) for mask in masks.values())
    expected = torch.arange(8)[None] <= torch.tensor([6, 7])[:, None]
    assert tuple(masks) == RELATIONS
    torch.testing.assert_close(total, expected.to(torch.int64))


def test_completed_builder_identifies_replicated_native_head():
    result = json.loads((ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_boundary_style_layer8_builder.json").read_text())
    assert result["hypotheses"]["B0_edge_partition_recomposes"]["supported"]
    assert result["hypotheses"]["B4_builder_head_identity_replicates"]["stable_folds"] == 3
    assert all(row["local_projection_share"]["head_2"] > .98 for row in result["results"])
