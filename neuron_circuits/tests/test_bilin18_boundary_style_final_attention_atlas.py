import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_bilin18_boundary_style_final_attention_atlas import RELATIONS,relation_masks  # noqa: E402

def test_final_attention_atlas_is_locked_before_results():
    config=json.loads((ROOT/"configs/bilin18_boundary_style_final_attention_atlas.json").read_text())
    assert config["status"].startswith("locked-")
    assert config["quadrature_points"]>=32

def test_final_source_relations_partition_prefix():
    masks=relation_masks(8,torch.tensor([2]),torch.tensor([5]),7); total=sum(v.long() for v in masks.values()); torch.testing.assert_close(total,torch.ones_like(total)); assert tuple(masks)==RELATIONS

def test_after_boundary_is_a_distinct_nonboundary_relation():
    masks=relation_masks(8,torch.tensor([2]),torch.tensor([5]),7)
    assert masks["boundary"][0,5]
    assert masks["after_boundary"][0,6]
    assert not masks["boundary"][0,6]

def test_relation_suffixes_do_not_alias():
    names=[f"h2_{relation}" for relation in RELATIONS]
    boundary=[name for name in names if name=="h2_boundary"]
    assert boundary==["h2_boundary"]
    assert "h2_after_boundary" not in boundary
