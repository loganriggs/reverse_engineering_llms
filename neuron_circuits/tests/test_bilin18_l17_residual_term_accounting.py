import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_term_accounting import response_terms
from bilin18_compiled_program import forward_normalized
from run_bilin18_l8_context_role_compression import B

def test_six_term_response_replays_polynomial_with_nonzero_mean_and_offsets():
    torch.manual_seed(132);x=torch.randn(5,9,dtype=torch.float64);z=torch.randn_like(x);mu=torch.randn(9,dtype=x.dtype);q=torch.linalg.qr(torch.randn(9,9,dtype=x.dtype)).Q;L=torch.randn(13,9,dtype=x.dtype);R=torch.randn_like(L);D=torch.randn(9,13,dtype=x.dtype)
    p={k:torch.randn(*shape,dtype=x.dtype) for k,shape in {'reads':(9,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,9),'bias':(9,)}.items()}
    def error(a):return B(a,a,L,R,D)-forward_normalized(a,p)
    terms=response_terms(x,z,mu,q[:,:2],q[:,2:5],q[:,:6],L,R,D,p)
    torch.testing.assert_close(terms.sum(1),error(z)-error(x))
    torch.testing.assert_close(torch.einsum('ntd,nud->ntu',terms,terms).sum((1,2)),(error(z)-error(x)).square().sum(1))
