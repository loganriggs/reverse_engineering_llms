import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_residual_restoration import score_gradients,fisher_maps

def test_score_gradient_includes_rms_and_softcap():
    torch.manual_seed(96);x=torch.randn(2,7,dtype=torch.float64,requires_grad=True);w=torch.randn(13,7,dtype=torch.float64);ids=torch.tensor([[1,4],[3,8]])
    analytic,_=score_gradients(x,w,ids)
    z=x/(x.square().mean(-1,keepdim=True)+torch.finfo(x.dtype).eps).sqrt();logp=(30*torch.tanh((z@w.T)/30)).log_softmax(-1)
    for row in range(2):
        for j in range(2):
            grad=torch.autograd.grad(logp[row,ids[row,j]],x,retain_graph=True)[0][row]
            torch.testing.assert_close(analytic[row,j],grad)

def test_full_rank_fisher_reconstruction_and_nested_idempotence():
    torch.manual_seed(97);e=torch.randn(25,7,dtype=torch.float64);g=torch.randn(100,7,dtype=torch.float64);maps,_=fisher_maps(e,g,[2,7])
    for encoder,decoder in maps.values():torch.testing.assert_close(encoder@decoder,torch.eye(len(encoder)),atol=1e-5,rtol=1e-5)
    en,de=maps[7];torch.testing.assert_close((e.float()@en.T)@de.T,e.float(),atol=1e-5,rtol=1e-5)
