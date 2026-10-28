import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_causal_transfer import mediation_state

def test_oracle_change_patch_recovers_target_and_cancels_intact_offsets():
    torch.manual_seed(102);changed,n0,n1,c0,c1,bias=[torch.randn(7,11,dtype=torch.float64) for _ in range(6)]
    oracle_delta=(n1-c1)-(n0-c0)
    torch.testing.assert_close(mediation_state(changed,n0,c1-c0,oracle_delta,bias),changed+n1+bias)
    offset=torch.randn_like(c0)
    torch.testing.assert_close((c1+offset)-(c0+offset),c1-c0)
