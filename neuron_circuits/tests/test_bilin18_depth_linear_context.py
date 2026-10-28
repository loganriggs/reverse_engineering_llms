import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_depth_linear_context import fit_ridge

def test_ridge_linear_response_is_integrable_and_recovers_linear_map():
    torch.manual_seed(91);x=torch.randn(40,7,dtype=torch.float64);w=torch.randn(7,5,dtype=torch.float64)
    fitted=fit_ridge(x,x@w,1e-9);torch.testing.assert_close(fitted,w,atol=1e-7,rtol=1e-7)
    a,b,c=torch.randn(3,7,dtype=torch.float64)
    torch.testing.assert_close((b-a)@fitted+(c-b)@fitted+(a-c)@fitted,torch.zeros(5,dtype=torch.float64),atol=1e-12,rtol=0)

def test_strict_compact_response_matches_compact_state_difference():
    from run_bilin18_depth_compact_input import compact_response
    from run_bilin18_l8_context_role_compression import B
    torch.manual_seed(92);d=torch.float64
    x=torch.randn(10,7,dtype=d);y=torch.randn(10,7,dtype=d);mu=torch.randn(7,dtype=d);q=torch.linalg.qr(torch.randn(7,3,dtype=d)).Q
    L=torch.randn(11,7,dtype=d);R=torch.randn(11,7,dtype=d);D=torch.randn(5,11,dtype=d)
    def state(z):
        p=mu+((z-mu)@q)@q.T;return B(p,p,L,R,D)
    torch.testing.assert_close(compact_response(x,y,mu,q,L,R,D)['strict'],state(y)-state(x))
    orth=torch.randn_like(x);orth=orth-(orth@q)@q.T
    torch.testing.assert_close(state(x+orth),state(x))
