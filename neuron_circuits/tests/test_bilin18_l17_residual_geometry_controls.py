import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_geometry_controls import match_norm

def test_norm_matching_preserves_direction_and_each_site_magnitude():
    torch.manual_seed(46)
    direction=torch.randn(8,13,dtype=torch.float64)
    reference=torch.randn_like(direction)*torch.arange(1,9,dtype=torch.float64)[:,None]
    result=match_norm(direction,reference)
    torch.testing.assert_close(result.norm(dim=-1),reference.norm(dim=-1))
    torch.testing.assert_close(result/result.norm(dim=-1,keepdim=True),direction/direction.norm(dim=-1,keepdim=True))
