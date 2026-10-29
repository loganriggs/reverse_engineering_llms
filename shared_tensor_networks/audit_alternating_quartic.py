"""Independent autograd update checks, coefficient-moment replay, and symmetry checks."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from audit_fixed_degree_reuse import coefficients,inner
from run_alternating_quartic import gradient,step

torch.set_num_threads(1)
root=Path(__file__).parent;out=root/'results/alternating_quartic'
assert json.loads((out/'completed.json').read_text())['fits']==36
protocol=json.loads((out/'protocol.json').read_text())
assert hashlib.sha256((out/'source.py').read_bytes()).hexdigest()==protocol['source_sha256']
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)))
x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
tx,tw=torch.tensor(x),torch.tensor(w);records=[];max_grad=0.;max_step=0.
for p in sorted(out.glob('s*_lr*.json')):
    r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));assert r['failure'] is None
    target=(x@d['teacher'].T).prod(1)/float(d['teacher_scale']);tt=torch.tensor(target);scale=float(d['student_scale'])
    for array in [d['initial'],d['student']]:
        a=torch.tensor(array,requires_grad=True);loss=.5*(tw*((tx@a.T).prod(1)/scale-tt).square()).sum()
        g=torch.autograd.grad(loss,a)[0].numpy();analytic=gradient(array,x,w,target,scale)
        err=float(np.linalg.norm(g-analytic)/max(1,np.linalg.norm(g)));assert err<1e-11;max_grad=max(max_grad,err)
    # Five complete sweeps independently recomputed through autograd, including order.
    a=d['initial'].copy();b=torch.tensor(a,requires_grad=True);rng=np.random.default_rng(r['seed']+201005)
    for k in range(5):
        order=None if r['mode']=='simultaneous' else list(range(4)) if r['mode']=='forward' else list(range(3,-1,-1)) if r['mode']=='reverse' else rng.permutation(4)
        a=step(a,x,w,target,scale,r['lr'],order)
        for i in ([None] if order is None else order):
            loss=.5*(tw*((tx@b.T).prod(1)/scale-tt).square()).sum();g=torch.autograd.grad(loss,b)[0]
            with torch.no_grad():
                if i is None:b-=r['lr']*g
                else:b[i]-=r['lr']*g[i]
    err=float(np.max(abs(a-b.detach().numpy())));assert err<1e-10;max_step=max(max_step,err)
    tc=coefficients(d['teacher'],float(d['teacher_scale']));sc=coefficients(d['student'],scale)
    delta={k:sc.get(k,0)-tc.get(k,0) for k in tc.keys()|sc.keys()};mse=inner(delta,delta)
    assert abs(mse-r['curves'][-1]['population_mse'])<1e-9*max(1,mse)
    assert r['block_gradient_components']==4*round(30/r['lr'])
    if r['mode']=='simultaneous':np.testing.assert_allclose(d['student'],np.broadcast_to(d['student'][0],(4,3)),atol=1e-12,rtol=1e-12)
    records.append(dict(name=r['name'],mse=mse,artifact_sha256=hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()))
result=dict(status='passed',fits=36,max_relative_gradient_error=max_grad,max_five_sweep_error=max_step,records=records,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'})
