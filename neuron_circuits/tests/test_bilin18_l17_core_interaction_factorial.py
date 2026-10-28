import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l17_residual_term_accounting import residual_bilinear

def test_quadratic_partition_and_basis_rotation_invariance():
    torch.manual_seed(270);dtype=torch.float64;x=torch.randn(7,9,dtype=dtype);z=torch.randn_like(x);mu=torch.randn(9,dtype=dtype);q=torch.linalg.qr(torch.randn(9,3,dtype=dtype)).Q;rot=torch.linalg.qr(torch.randn(3,3,dtype=dtype)).Q;L=torch.randn(13,9,dtype=dtype);R=torch.randn_like(L);D=torch.randn(9,13,dtype=dtype)
    p={k:torch.randn(*shape,dtype=dtype) for k,shape in {'reads':(9,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,9),'bias':(9,)}.items()}
    t=quadratic_terms(x,z,mu,q,L,R,D,p)
    torch.testing.assert_close(t.sum(1),residual_bilinear(x+z-2*mu,z-x,L,R,D,p))
    torch.testing.assert_close(t,quadratic_terms(x,z,mu,q@rot,L,R,D,p))
