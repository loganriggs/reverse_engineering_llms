import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_partner_objective_mixture import combined_loss

def test_family_normalized_loss_rescaling_and_gradient_weights():
    a=torch.tensor(4.,dtype=torch.float64,requires_grad=True);b=torch.tensor(12.,dtype=torch.float64,requires_grad=True)
    loss=combined_loss({'actual':a,'angle':b},{'actual':2.,'angle':6.},{'actual':.5,'angle':.5});assert loss.item()==2.
    ga,gb=torch.autograd.grad(loss,(a,b));torch.testing.assert_close(ga,torch.tensor(.25,dtype=a.dtype));torch.testing.assert_close(gb,torch.tensor(1/12,dtype=b.dtype))
    assert combined_loss({'actual':a*9,'angle':b},{'actual':18.,'angle':6.},{'actual':.5,'angle':.5}).item()==loss.item()
    assert combined_loss({'actual':a},{'actual':2.},{'actual':1.,'angle':0.}).item()==2.
