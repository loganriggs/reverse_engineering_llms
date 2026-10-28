import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_intervention_input_metric import metric_form,role_form,response,scalar
from run_bilin18_residual_input_forms import quadratic
from run_bilin18_l8_source_span_response import source_form


def fixture():
    torch.manual_seed(42)
    q=torch.randn(9,9,dtype=torch.float64);q=(q+q.T)/2
    c=torch.randn(9,9,dtype=torch.float64);c=c@c.T+torch.eye(9)
    return q,c


def test_full_rank_metrics_recover_same_quadratic():
    q,c=fixture()
    torch.testing.assert_close(metric_form(q,c,9),q)
    torch.testing.assert_close(role_form(q,c,torch.eye(9,dtype=torch.float64),9),q)


def test_response_equals_scalar_difference_for_symmetric_decoder():
    q,c=fixture();a=metric_form(q,c,4)
    x=torch.randn(20,9,dtype=torch.float64)+1;y=torch.randn_like(x)-2;mean=x.mean(0)
    torch.testing.assert_close(response(x,y,q,a,mean),scalar(y,q,a,mean,x)-scalar(x,q,a,mean,x))
    torch.testing.assert_close(response(x,y,q,q,mean),quadratic(y,q)-quadratic(x,q))


def test_unsymmetric_response_can_fail_cycle_but_state_decoder_cannot():
    q,c=fixture();a=role_form(q,c,torch.eye(9,dtype=torch.float64),2)
    x=torch.randn(20,9,dtype=torch.float64);y=torch.randn_like(x);z=torch.randn_like(x);mean=x.mean(0)
    def cycle(secant):
        return response(x,y,q,a,mean,secant)+response(y,z,q,a,mean,secant)+response(z,x,q,a,mean,secant)
    assert cycle(True).square().sum()>1e-6
    assert cycle(False).abs().max()<1e-10
    assert torch.linalg.matrix_rank((a+a.T)/2)<=4


def test_source_span_preserves_all_changes_inside_span():
    q,_=fixture();basis=torch.linalg.qr(torch.randn(9,2,dtype=torch.float64)).Q
    a=source_form(q,basis);x=torch.randn(20,9,dtype=torch.float64)
    y=x+torch.randn(20,2,dtype=torch.float64)@basis.T
    mean=x.mean(0)
    torch.testing.assert_close(response(x,y,q,a,mean),quadratic(y,q)-quadratic(x,q))
    assert torch.linalg.matrix_rank(a)<=4
