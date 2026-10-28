import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_kl_intercept import distillation_kl

def test_teacher_to_prediction_kl_and_gradient():
    torch.manual_seed(278);dt=torch.float64;t=torch.randn(3,11,dtype=dt);lp=t.log_softmax(-1);s=torch.randn_like(t,requires_grad=True)
    torch.testing.assert_close(distillation_kl(t,lp),torch.zeros(3,dtype=dt))
    loss=distillation_kl(s,lp).sum();g=torch.autograd.grad(loss,s)[0]
    torch.testing.assert_close(g,s.softmax(-1)-lp.exp())
    assert torch.autograd.gradcheck(lambda x:distillation_kl(x,lp),(s,))
