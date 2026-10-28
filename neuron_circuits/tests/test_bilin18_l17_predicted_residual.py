import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_predicted_residual import fit_affine

def test_affine_ridge_intercept_and_normal_equations():
    torch.manual_seed(84);x=torch.randn(60,8)+2;y=torch.randn(60,5)+3;w,b=fit_affine(x,y,.1);xc=x-x.mean(0);yc=y-y.mean(0);g=xc.T@xc;lam=.1*g.trace()/8
    torch.testing.assert_close((g+lam*torch.eye(8))@w,xc.T@yc,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close((x@w+b).mean(0),y.mean(0))
