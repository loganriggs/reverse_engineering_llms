"""Differentiate expanded polynomial coefficients independently of autograd."""
import json
from pathlib import Path
import numpy as np
import torch
from audit_fixed_degree_reuse import coefficients,inner
from run_fixed_degree_reuse import quadrature,polynomial

def main():
    torch.set_num_threads(1);out=Path(__file__).parent/'results/fixed_degree_reuse';xx,ww=quadrature();x=torch.tensor(xx);w=torch.tensor(ww);worst=0.;count=0
    for f in sorted(out.glob('t*_s*.npz')):
        z=np.load(f);a=z['initial'];scale=float(z['student_scale']);tc=coefficients(z['teacher'],float(z['teacher_scale']));sc=coefficients(a,scale);delta={p:sc.get(p,0)-tc.get(p,0) for p in set(sc)|set(tc)};expected=np.zeros_like(a);r=4//len(a)
        for i in range(len(a)):
            for j in range(3):
                factors=np.repeat(a,r,axis=0);factors[i*r]=np.eye(3)[j];dc=coefficients(factors,scale);expected[i,j]=r*inner(delta,dc)
        at=torch.tensor(a,requires_grad=True);target=polynomial(x,torch.tensor(z['teacher']))/float(z['teacher_scale']);ell=.5*(w@(polynomial(x,at)/scale-target).square());actual=torch.autograd.grad(ell,at)[0].numpy();err=float(np.max(abs(actual-expected))/max(1,np.max(abs(expected))));assert err<1e-10;worst=max(worst,err);count+=1
    (out/'gradient_check.json').write_text(json.dumps(dict(status='passed',fits=count,max_relative_error=worst),indent=2)+'\n');print(count,worst)
if __name__=='__main__':main()
