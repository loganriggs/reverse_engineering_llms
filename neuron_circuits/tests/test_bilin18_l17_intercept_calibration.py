import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l17_intercept_calibration import fit_output_constant

def test_constant_fit_optimality_and_response_invariance():
    torch.manual_seed(277);target=torch.randn(17,7,dtype=torch.float64);pred=torch.randn_like(target);b=fit_output_constant(target,pred);delta=torch.randn_like(b)
    base=(target-pred-b).square().sum();shifted=(target-pred-b-delta).square().sum()
    torch.testing.assert_close(shifted-base,len(target)*delta.square().sum())
    torch.testing.assert_close((pred[1:]+b)-(pred[:-1]+b),pred[1:]-pred[:-1])
