import torch
from run_bilin18_l17_response_regularized import stats,response_loss

def test_eigen_normal_solve_matches_lstsq_and_optimality_identity():
    torch.manual_seed(19);x=torch.randn(211,13,dtype=torch.float64);y=torch.randn(211,17,dtype=torch.float64);g,r,e=stats(x,y);ev,q=torch.linalg.eigh(g);w=(q/ev)@(q.T@r);direct=torch.linalg.lstsq(x,y).solution
    assert torch.allclose(w,direct,rtol=1e-10,atol=1e-10)
    d=torch.randn_like(w)*.1;gap=response_loss(w+d,g,r,e)-response_loss(w,g,r,e)
    assert torch.allclose(gap,(d*(g@d)).sum()/e,atol=1e-12,rtol=1e-10)
