import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_response_capacity import response_solutions

def test_response_projection_matches_lstsq_and_handles_null_feature():
    torch.manual_seed(114);x=torch.randn(50,7,dtype=torch.float64);x[:,-1]=x[:,0]+x[:,1];y=torch.randn(50,5,dtype=torch.float64)
    models,audit=response_solutions(x,y,[0,.1]);reference=torch.linalg.lstsq(x,y,driver='gelsd').solution
    torch.testing.assert_close(x@models[0],x@reference,rtol=1e-7,atol=1e-7)
    assert audit['rank']==6 and audit['fits']['0']['normal_relative_residual']<1e-10
    assert audit['fits']['0']['training_fidelity']>=audit['fits']['0.1']['training_fidelity']
