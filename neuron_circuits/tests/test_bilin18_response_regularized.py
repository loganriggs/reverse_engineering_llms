import torch
from run_bilin18_l17_response_regularized import response_loss,stats

def test_sufficient_statistics_match_explicit_loss_and_gradient():
    torch.manual_seed(4);df=torch.randn(97,7,dtype=torch.float64);target=torch.randn(97,11,dtype=torch.float64);w=torch.randn(7,11,dtype=torch.float64,requires_grad=True)
    fast=response_loss(w,*stats(df,target));explicit=(df@w-target).square().sum()/target.square().sum()
    assert torch.allclose(fast,explicit,atol=1e-12,rtol=1e-12)
    g1=torch.autograd.grad(fast,w,retain_graph=True)[0];g2=torch.autograd.grad(explicit,w)[0]
    assert torch.allclose(g1,g2,atol=1e-12,rtol=1e-12)
    a=stats(df[:43],target[:43]);b=stats(df[43:],target[43:]);combined=[x+y for x,y in zip(a,b)]
    assert torch.allclose(response_loss(w,*combined),explicit,atol=1e-12,rtol=1e-12)
