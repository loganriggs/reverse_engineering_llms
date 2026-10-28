import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_context_role_compression import B,operator_metric,response_rules,state_rules

def test_operator_metric_equals_jacobian_norm():
    torch.manual_seed(3)
    L=torch.randn(11,7,dtype=torch.double);R=torch.randn_like(L);D=torch.randn(5,11,dtype=torch.double);v=torch.randn(7,dtype=torch.double)
    J=D@((R@v)[:,None]*L+(L@v)[:,None]*R)
    torch.testing.assert_close(v@operator_metric(L,R,D)@v,J.square().sum())

def test_integrable_rules_are_state_differences_for_oblique_encoder():
    torch.manual_seed(8)
    L=torch.randn(11,7,dtype=torch.double);R=torch.randn_like(L);D=torch.randn(5,11,dtype=torch.double)
    x,y,z=[torch.randn(9,7,dtype=torch.double) for _ in range(3)];mean=torch.randn(7,dtype=torch.double)
    E=torch.randn(3,7,dtype=torch.double);A=torch.randn(7,3,dtype=torch.double)
    xy=response_rules(x,y,mean,E,A,L,R,D);sx=state_rules(x,mean,E,A,L,R,D);sy=state_rules(y,mean,E,A,L,R,D)
    yz=response_rules(y,z,mean,E,A,L,R,D);zx=response_rules(z,x,mean,E,A,L,R,D)
    for rule in sx:
        torch.testing.assert_close(xy[rule],sy[rule]-sx[rule])
        torch.testing.assert_close(xy[rule]+yz[rule]+zx[rule],torch.zeros_like(xy[rule]),atol=1e-10,rtol=0)
    assert (xy['context_only']+yz['context_only']+zx['context_only']).norm()>1e-3
    identity=torch.eye(7,dtype=torch.double);full=response_rules(x,y,mean,identity,identity,L,R,D)
    for prediction in full.values():torch.testing.assert_close(prediction,B(y,y,L,R,D)-B(x,x,L,R,D))
