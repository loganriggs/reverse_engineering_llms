import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_bilin18_l8_context_response_geometry import source_jacobian

def test_source_derivative_matches_autograd_with_rms_radial_term():
    torch.manual_seed(3)
    pre=torch.randn(2,7,dtype=torch.double);left=torch.randn(11,7,dtype=torch.double);right=torch.randn_like(left);down=torch.randn(5,11,dtype=torch.double);source=torch.randn(7,3,dtype=torch.double)
    J=source_jacobian(pre,left,right,down,source)
    for n in range(2):
        def fn(x):
            z=torch.nn.functional.rms_norm(x,(7,))
            return down@((left@z)*(right@z))
        expected=torch.autograd.functional.jacobian(fn,pre[n])@source
        torch.testing.assert_close(J[n],expected,atol=1e-11,rtol=1e-11)
