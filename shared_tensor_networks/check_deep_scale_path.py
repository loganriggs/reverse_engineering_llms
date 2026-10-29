"""Verify simultaneous parameter scaling with fixed calibration buffers."""
import json
from pathlib import Path
import torch
from deep_multitask import DeepMultiTask


def main():
    torch.set_num_threads(1);rows=[];g=torch.Generator().manual_seed(127000)
    x=torch.randn(17,3,generator=g,dtype=torch.float64);y=torch.randn(17,2,generator=g,dtype=torch.float64)
    for depth in [2,3,4,5]:
        m=DeepMultiTask(3,2,2,depth,True,128000+depth);m.calibrate(x)
        t=torch.tensor(.9,dtype=torch.float64,requires_grad=True);base=m(x).detach()
        # Trunk parameter degree e_l=2e_(l-1)+2; readout is quadratic
        # in the final trunk with one extra linear W factor.
        D=2**(depth+1)-3
        pred=torch.func.functional_call(m,{k:v*t for k,v in m.named_parameters()},(x,))
        torch.testing.assert_close(pred,t**D*base,rtol=1e-10,atol=1e-10)
        loss=.5*(pred-y).square().mean();grad,=torch.autograd.grad(loss,t)
        a=base.square().mean();c=(base*y).mean();expected=D*a*t**(2*D-1)-D*c*t**(D-1)
        torch.testing.assert_close(grad,expected,rtol=1e-10,atol=1e-10)
        rows.append(dict(depth=depth,input_degree=2**depth,parameter_degree=D,gradient_error=float(abs(grad-expected).detach())))
    out=Path(__file__).parent/'results/coverage_training/scale_path_checks.json'
    out.write_text(json.dumps(dict(status='passed',seed=127000,rows=rows),indent=2)+'\n');print(rows)


if __name__=='__main__':main()
