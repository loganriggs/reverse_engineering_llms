import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_response_supervision import joint_fit

def test_joint_fit_response_intercept_cancels_and_beta_zero_ignores_responses():
    torch.manual_seed(109);f=torch.randn(30,5);y=torch.randn(30,4);df=torch.randn(80,5);dy=torch.randn(80,4)
    w,_,_,res=joint_fit(f,y,df,dy,0,.1);w2,_,_,_=joint_fit(f,y,df*2,dy*3,0,.1)
    torch.testing.assert_close(w,w2);assert res<1e-10
    torch.testing.assert_close(f.mean(0)@w[:-1]+w[-1],y.mean(0),atol=1e-6,rtol=1e-5)
    wj,_,_,res=joint_fit(f,y,df,dy,1,.1);assert res<1e-10
    torch.testing.assert_close(f.mean(0)@wj[:-1]+wj[-1],y.mean(0),atol=1e-6,rtol=1e-5)
