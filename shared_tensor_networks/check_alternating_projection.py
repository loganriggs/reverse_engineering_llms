"""Projection equivalence survives layerwise alternation; reciprocal gauge identities."""
import itertools
from pathlib import Path
import numpy as np
import torch
from experiment import Network
from run_bilinear_alternation import evaluate,gradients,update,blocks
from run_svhn_standard import write,digest

torch.set_num_threads(1)
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)))
x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
rows=[]
for seed in [211001,211002,211003]:
    teacher=Network(3,3,3,False,seed+100);target=teacher(torch.tensor(x)).detach().numpy();target/=np.sqrt(w@target**2)
    m=Network(3,3,3,True,seed+200);base={k:v.detach().numpy().copy() for k,v in m.named_parameters()};scale=np.sqrt(w@evaluate(base,x)**2)
    for method in ['joint','forward','reverse']:
        tied={k:v.copy() for k,v in base.items()};free={k:v.copy() for k,v in base.items()};free.update(AR=free['A'].copy(),BR=free['B'].copy())
        worst=0.;gauge_error=0.;balance_error=0.
        for step in range(20):
            g=gradients(tied,x,w,target,scale)
            for a,b in [('A','B'),('C','D')]:
                # Rowwise exact gauge derivative identity.
                error=np.max(abs((tied[a]*g[a]).sum(1)-(tied[b]*g[b]).sum(1)))
                gauge_error=max(gauge_error,float(error));assert error<1e-10
                old=(tied[a]**2).sum(1)-(tied[b]**2).sum(1)
                eta=.01/(2 if a=='A' else 1)
                new=((tied[a]-eta*g[a])**2).sum(1)-((tied[b]-eta*g[b])**2).sum(1)
                expected=eta**2*((g[a]**2).sum(1)-(g[b]**2).sum(1))
                error=np.max(abs((new-old)-expected));assert error<1e-10;balance_error=max(balance_error,float(error))
            tied=update(tied,x,w,target,scale,.01,method,'corrected')
            groups=blocks(free);schedule=[sum(groups,[])] if method=='joint' else groups if method=='forward' else groups[::-1]
            for group in schedule:
                gf=gradients(free,x,w,target,scale)
                for k in group:free[k]-=.01*gf[k]
                for a,b in [('A','AR'),('B','BR')]:
                    average=(free[a]+free[b])/2;free[a]=average.copy();free[b]=average.copy()
            error=max(float(np.max(abs(tied[k]-free[k]))) for k in tied);assert error<1e-10;worst=max(worst,error)
        rows.append(dict(seed=seed,method=method,max_parameter_error=worst,max_gauge_derivative_error=gauge_error,max_discrete_balance_error=balance_error))
root=Path(__file__).parent;out=root/'results/alternating_projection';out.mkdir(exist_ok=True)
write(out/'checks.json',dict(status='passed',steps=20,rows=rows,source_sha256=digest(Path(__file__)),dependency_sha256=digest(root/'run_bilinear_alternation.py')))
print(rows)
