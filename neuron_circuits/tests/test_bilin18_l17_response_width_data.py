import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_response_width_data import width_features
from run_bilin18_l17_residual_feature_objective import raw_features

def test_width32_replays_parent_and_width64_adds_nested_products():
    torch.manual_seed(117);x=torch.randn(4,1152);basis=torch.eye(1152);mean=torch.randn(1152)
    a=width_features(x,basis,mean,32);b=width_features(x,basis,mean,64)
    torch.testing.assert_close(a,raw_features(x,basis,mean,'quadratic1152'))
    ij=torch.triu_indices(64,64);keep=(ij[0]<32)&(ij[1]<32)
    torch.testing.assert_close(a[:,624:],b[:,624:][:,keep]);assert b.shape==(4,2704)
