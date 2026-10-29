"""Repeated-squaring size certificate and explicit scalar gradient-flow fixture."""
import json
from pathlib import Path
import torch

def main():
    rows=[]
    for depth in range(1,7):
        m=2**depth;t=torch.tensor(.8,dtype=torch.float64,requires_grad=True);h=t
        for _ in range(depth):h=h*h
        loss=.5*h*h;grad=torch.autograd.grad(loss,t,create_graph=True)[0];hess=torch.autograd.grad(grad,t)[0]
        torch.testing.assert_close(grad,m*t**(2*m-1),rtol=1e-12,atol=0)
        torch.testing.assert_close(hess,m*(2*m-1)*t**(2*m-2),rtol=1e-12,atol=0)
        z=torch.full((m,),.8,dtype=torch.float64,requires_grad=True);free_loss=.5*z.prod().square();g=torch.autograd.grad(free_loss,z)[0]
        torch.testing.assert_close(g.sum(),grad,rtol=1e-12,atol=0)
        # Closed-form zero-target ordinary tied gradient flow, with dot(t)=-m*t^(2m-1).
        time=torch.tensor(.13,dtype=torch.float64,requires_grad=True);solution=(.8**(-(2*m-2))+m*(2*m-2)*time)**(-1/(2*m-2))
        velocity=torch.autograd.grad(solution,time)[0]
        torch.testing.assert_close(velocity,-m*solution**(2*m-1),rtol=1e-11,atol=0)
        rows.append(dict(depth=depth,degree=m,minimum_multiplication_gates=depth,minimum_formula_gates=m-1,local_rlct_zero_target=1/(2*m),gradient=float(grad.detach()),hessian=float(hess.detach())))
    out=Path(__file__).parent/'results/reuse_flatness';out.mkdir(exist_ok=False);(out/'checks.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2)+'\n');print('Six repeated-squaring degree/derivative/flow checks passed')
if __name__=='__main__':main()
