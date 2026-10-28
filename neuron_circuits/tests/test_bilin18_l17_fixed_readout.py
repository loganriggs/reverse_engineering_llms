import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_fixed_readout import read_features,ridge_fit

def test_adapter_uses_only_existing_reads_and_has_linear_response():
    torch.manual_seed(280);dt=torch.float64;q=torch.linalg.qr(torch.randn(9,5,dtype=dt)).Q;p={'mean':torch.randn(9,dtype=dt),'core':q[:,:3],'partner':q[:,3:]};center=torch.randn(5,dtype=dt);scale=torch.rand(5,dtype=dt)+.2;w=torch.randn(5,9,dtype=dt);x=torch.randn(6,9,dtype=dt);z=torch.randn_like(x);ambient=(q/scale)@w
    torch.testing.assert_close(read_features(z,p,center,scale)@w-read_features(x,p,center,scale)@w,(z-x)@ambient)
    torch.testing.assert_close(read_features(x,p,center,scale)@w,(x-p['mean'])@ambient-(center/scale)@w)

def test_ridge_satisfies_regularized_normal_equations():
    torch.manual_seed(281);x=torch.randn(29,7,dtype=torch.float64);y=torch.randn(29,5,dtype=torch.float64);g=x.T@x
    for alpha in [0.,.1,10.]:
        w=ridge_fit(x,y,alpha);torch.testing.assert_close((g+alpha*g.diag().mean()*torch.eye(7))@w,x.T@y)
