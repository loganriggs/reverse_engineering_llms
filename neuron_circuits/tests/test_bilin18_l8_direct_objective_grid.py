import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_direct_objective_grid import qvalues,qresponse

def test_finite_response_is_difference_and_cycle_closes():
    torch.manual_seed(18)
    u=torch.randn(3,7,4,dtype=torch.double);w=torch.randn(3,4,dtype=torch.double)
    x,y,z=[torch.randn(11,7,dtype=torch.double) for _ in range(3)]
    def response(a,b):return qresponse(b-a,(a+b)/2,u,w)
    torch.testing.assert_close(response(x,y),qvalues(y,u,w)-qvalues(x,u,w))
    torch.testing.assert_close(response(x,y)+response(y,z)+response(z,x),torch.zeros(11,3,dtype=torch.double),atol=1e-12,rtol=0)

def test_value_difference_loss_equals_mean_calibrated_error():
    torch.manual_seed(3)
    error=torch.randn(19,4,dtype=torch.double)+7
    paired=(error[:,None]-error[None,:]).square().mean((0,1))/2
    torch.testing.assert_close(paired,(error-error.mean(0)).square().mean(0))

def test_response_gradient_matches_finite_difference():
    torch.manual_seed(9)
    u=torch.randn(2,5,3,dtype=torch.double,requires_grad=True)
    w=torch.randn(2,3,dtype=torch.double,requires_grad=True)
    d,m=[torch.randn(4,5,dtype=torch.double) for _ in range(2)]
    assert torch.autograd.gradcheck(lambda a,b:qresponse(d,m,a,b),(u,w))
