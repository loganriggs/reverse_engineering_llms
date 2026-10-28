import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_readout_temperature import temperature_scores
from run_bilin18_l17_residual_restoration import score_gradients

def test_temperature_score_matches_autograd_and_native_formula():
    torch.manual_seed(73);x=torch.randn(2,7,dtype=torch.float64,requires_grad=True);w=torch.randn(13,7,dtype=torch.float64);ids=torch.tensor([[1,4],[2,6]])
    for t in [.5,1.,2.]:
        analytic,p=temperature_scores(x,w,ids,t)
        lp=(30*torch.tanh((torch.nn.functional.rms_norm(x,(7,))@w.T)/30)/t).log_softmax(-1)
        for b in range(2):
            for k in range(2):
                grad=torch.autograd.grad(lp[b,ids[b,k]],x,retain_graph=True)[0][b]
                torch.testing.assert_close(grad,analytic[b,k])
        if t==1.:
            old,oldp=score_gradients(x,w,ids);torch.testing.assert_close(analytic,old);torch.testing.assert_close(p,oldp)
