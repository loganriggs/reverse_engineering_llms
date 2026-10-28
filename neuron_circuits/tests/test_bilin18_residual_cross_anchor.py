import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bilin18_residual_cross_program import compile_program,forward,intact_forward,change
from bilin18_compiled_program import forward_normalized
from run_bilin18_l8_context_role_compression import B

def test_anchor_matches_mean_without_changing_response():
    torch.manual_seed(276);dt=torch.float64;mu=torch.randn(9,dtype=dt);q=torch.linalg.qr(torch.randn(9,9,dtype=dt)).Q;L=torch.randn(13,9,dtype=dt);R=torch.randn_like(L);D=torch.randn(9,13,dtype=dt)
    p={k:torch.randn(*shape,dtype=dt) for k,shape in {'reads':(9,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,9),'bias':(9,)}.items()}
    pr=compile_program(mu,q[:,:3],q[:,3:5],L,R,D,p);pr['anchor']=(B(mu[None],mu[None],L,R,D)-forward_normalized(mu[None],p))[0]
    torch.testing.assert_close(forward_normalized(mu[None],p)+intact_forward(mu[None],pr),B(mu[None],mu[None],L,R,D))
    x=torch.randn(4,9,dtype=dt);z=torch.randn_like(x);torch.testing.assert_close(intact_forward(z,pr)-intact_forward(x,pr),change(x,z,pr))
