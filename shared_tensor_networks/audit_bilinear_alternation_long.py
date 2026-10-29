"""Independent autograd block updates and Gaussian coefficient-moment final replay."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from experiment import Network,coefficients,gram
from run_bilinear_alternation_long import gradients,update
from run_svhn_standard import digest,write

torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/bilinear_alternation_long'
assert json.loads((out/'completed.json').read_text())['fits']==108
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
for name,sha in protocol['dependencies'].items():assert digest(root/name)==sha
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)))
x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1);tx,tw=torch.tensor(x),torch.tensor(w)

def load(parameters):
    m=Network(3,3,3,'AR' not in parameters,0)
    with torch.no_grad():
        for k,p in m.named_parameters():p.copy_(torch.tensor(parameters[k]))
    return m

records=[];gradient_error=0.;step_error=0.
for path in sorted(out.glob('*_0.*_*.json')):
    r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'))
    if r['failure']:continue
    states={prefix:{k[len(prefix)+1:]:d[k] for k in d.files if k.startswith(prefix+'_')} for prefix in ['teacher','initial','final']}
    # Scales have the same prefix; exclude those scalar metadata fields.
    for state in states.values():state.pop('scale',None)
    teacher=load(states['teacher']);scale=float(d['student_scale']);tscale=float(d['teacher_scale'])
    target=teacher(tx).detach()/tscale
    model=load(states['final']);loss=.5*(tw*(model(tx)/scale-target).square()).sum()
    ag=torch.autograd.grad(loss,tuple(model.parameters()));g=gradients(states['final'],x,w,target.numpy(),scale)
    error=max(float(np.linalg.norm(v.numpy()-g[k])/max(1,np.linalg.norm(v.numpy()))) for (k,p),v in zip(model.named_parameters(),ag));assert error<1e-10;gradient_error=max(gradient_error,error)
    c=coefficients(model);t=coefficients(teacher);keys=sorted(c.keys()|t.keys());delta=np.array([c.get(k,0)/scale-t.get(k,0)/tscale for k in keys]);mse=float(delta@gram(keys)@delta)
    assert abs(mse-r['curves'][-1]['mse'])<1e-9*max(1,mse)
    model=load(states['initial']);parameters=dict(model.named_parameters());p={k:v.copy() for k,v in states['initial'].items()}
    groups=[['A','B']+(['AR','BR'] if 'AR' in p else []),['C','D'],['w']]
    schedule=[sum(groups,[])] if r['method']=='joint' else groups if r['method']=='forward' else groups[::-1]
    for sweep in range(3):
        p=update(p,x,w,target.numpy(),scale,r['lr'],r['method'],r['metric'])
        for group in schedule:
            loss=.5*(tw*(model(tx)/scale-target).square()).sum();gg=torch.autograd.grad(loss,tuple(parameters.values()))
            with torch.no_grad():
                for (k,param),v in zip(parameters.items(),gg):
                    if k in group:param-=r['lr']*v/(2 if r['metric']=='corrected' and k in ['A','B'] else 1)
    error=max(float(np.max(abs(v.detach().numpy()-p[k]))) for k,v in parameters.items());assert error<1e-10;step_error=max(step_error,error)
    records.append(dict(name=r['name'],mse=mse,artifact_sha256=digest(path.with_suffix('.npz'))))
assert len(records)==108
write(out/'audit.json',dict(status='passed',fits=len(records),max_final_gradient_error=gradient_error,max_three_sweep_error=step_error,records=records,source_sha256=digest(Path(__file__))))
print('Audited',len(records),'gradient',gradient_error,'updates',step_error)
