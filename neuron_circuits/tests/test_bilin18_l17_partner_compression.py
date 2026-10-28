import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_partner_compression import cross_response,core_metric
from run_bilin18_l17_core_interaction_factorial import quadratic_terms

def test_full_partner_cross_and_metric_coordinate_covariance():
    torch.manual_seed(271);dt=torch.float64;x=torch.randn(7,9,dtype=dt);z=torch.randn_like(x);mu=torch.randn(9,dtype=dt);q=torch.linalg.qr(torch.randn(9,9,dtype=dt)).Q;c=q[:,:3];rot=torch.linalg.qr(torch.randn(3,3,dtype=dt)).Q;L=torch.randn(13,9,dtype=dt);R=torch.randn_like(L);D=torch.randn(9,13,dtype=dt)
    p={k:torch.randn(*shape,dtype=dt) for k,shape in {'reads':(9,4),'offset':(4,),'eigenvectors':(3,4,2),'eigenvalues':(3,2),'linear':(3,4),'outputs':(3,9),'bias':(9,)}.items()}
    torch.testing.assert_close(cross_response(x,z,mu,c,q[:,3:],L,R,D,p),quadratic_terms(x,z,mu,c,L,R,D,p)[:,1])
    m=core_metric(x,z,mu,c);assert torch.linalg.eigvalsh(m).min()>0
    torch.testing.assert_close(core_metric(x,z,mu,c@rot),rot.T@m@rot)
