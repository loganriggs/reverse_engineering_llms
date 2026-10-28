import torch


def test_two_factor_shapley_recomposes_endpoint():
    torch.manual_seed(0); worlds={(0,0):torch.randn(11),(1,0):torch.randn(11),(0,1):torch.randn(11),(1,1):torch.randn(11)}
    core=worlds[(0,0)]; injection=.5*((worlds[(1,0)]-worlds[(0,0)])+(worlds[(1,1)]-worlds[(0,1)])); bus=.5*((worlds[(0,1)]-worlds[(0,0)])+(worlds[(1,1)]-worlds[(1,0)]))
    assert torch.allclose(core+injection+bus,worlds[(1,1)],atol=1e-6,rtol=1e-6)
