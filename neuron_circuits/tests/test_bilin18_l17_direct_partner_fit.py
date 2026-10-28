import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_direct_partner_fit import partner_q
from run_bilin18_l17_partner_compression import cross_response

def test_partner_projection_gradient_and_constraints():
    torch.manual_seed(272);dt=torch.float64;x=torch.randn(3,7,dtype=dt);z=torch.randn_like(x);mu=torch.randn(7,dtype=dt);c=torch.linalg.qr(torch.randn(7,2,dtype=dt)).Q;L=torch.randn(8,7,dtype=dt);R=torch.randn_like(L);D=torch.randn(7,8,dtype=dt)
    p={k:torch.randn(*shape,dtype=dt) for k,shape in {'reads':(7,3),'offset':(3,),'eigenvectors':(2,3,2),'eigenvalues':(2,2),'linear':(2,3),'outputs':(2,7),'bias':(7,)}.items()}
    w=torch.randn(7,2,dtype=dt,requires_grad=True)
    fun=lambda w:cross_response(x,z,mu,c,partner_q(w,c),L,R,D,p).square().mean()
    assert torch.autograd.gradcheck(fun,(w,),eps=1e-6,atol=1e-4,rtol=1e-3)
    q=partner_q(w,c);torch.testing.assert_close(q.T@q,torch.eye(2,dtype=dt));torch.testing.assert_close(c.T@q,torch.zeros(2,2,dtype=dt),atol=1e-12,rtol=0)
