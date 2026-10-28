import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_kl_constant_audit import mean_gradient

def test_full_mean_gradient_invariant_to_chunk_size():
    torch.manual_seed(279);x=torch.randn(19,7,dtype=torch.float64);target=torch.randn(19,dtype=torch.float64);b=torch.randn(7,dtype=torch.float64,requires_grad=True)
    def loss(v,i,j):return (x[i:j]@v-target[i:j]).square()
    expected=2*x.T@(x@b-target)/len(x)
    for batch in [1,6,19]:torch.testing.assert_close(mean_gradient(loss,b,len(x),batch),expected)
