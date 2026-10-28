import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bilin18_residual_cross_program import compile_program,forward,change
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import mean_linear

def test_compiled_response_matches_native_polynomial_on_arbitrary_inputs():
    torch.manual_seed(275);dt=torch.float64;x=torch.randn(5,9,dtype=dt);z=torch.randn_like(x);mu=torch.randn(9,dtype=dt);q=torch.linalg.qr(torch.randn(9,9,dtype=dt)).Q;c=q[:,:3];v=q[:,3:5];L=torch.randn(13,9,dtype=dt);R=torch.randn_like(L);D=torch.randn(9,13,dtype=dt)
    p={k:torch.randn(*shape,dtype=dt) for k,shape in {'reads':(9,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,9),'bias':(9,)}.items()}
    pr=compile_program(mu,c,v,L,R,D,p);base=mean_linear(z-x,mu,L,R,D,p)+quadratic_terms(x,z,mu,c,L,R,D,p)[:,0];cross=cross_response(x,z,mu,c,v,L,R,D,p)
    torch.testing.assert_close(forward(z,pr)-forward(x,pr),base+cross)
    torch.testing.assert_close(forward(z,pr,False)-forward(x,pr,False),base)

    torch.testing.assert_close(change(x,z,pr),base+cross)
    torch.testing.assert_close(change(x,z,pr,False),base)
