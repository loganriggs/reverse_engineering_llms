import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_response_input_bases import residual_pullback
from bilin18_compiled_program import forward_normalized
from run_bilin18_l8_context_role_compression import B

def test_residual_pullback_matches_autograd_including_compiled_quadratic():
    torch.manual_seed(119);x=torch.randn(2,7,dtype=torch.float64,requires_grad=True);v=torch.randn(2,3,7,dtype=torch.float64);L=torch.randn(11,7,dtype=torch.float64);R=torch.randn_like(L);D=torch.randn(7,11,dtype=torch.float64)
    p={k:torch.randn(*shape,dtype=torch.float64) for k,shape in {'reads':(7,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,7),'bias':(7,)}.items()}
    expected=B(x,x,L,R,D)-forward_normalized(x,p);g=residual_pullback(x,v,L,R,D,p)
    for k in range(3):
        reference=torch.autograd.grad((expected*v[:,k]).sum(),x,retain_graph=True)[0];torch.testing.assert_close(g[:,k],reference)
