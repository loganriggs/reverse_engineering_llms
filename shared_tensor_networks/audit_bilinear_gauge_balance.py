"""Autograd Jacobian, exact coefficient moments, descent and linear-solve checks."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from experiment import Network,coefficients,gram
from run_bilinear_gauss_newton import jacobian,unpack
from run_bilinear_alternation import evaluate
from run_svhn_standard import digest,write

torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/bilinear_gauge_balance'
assert json.loads((out/'completed.json').read_text())['fits']==54
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)));x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1);tx=torch.tensor(x)
records=[];worst_j=0.;worst_linear=0.;worst_moment=0.
def model(p):
    m=Network(3,3,3,'AR' not in p,0)
    with torch.no_grad():
        for k,v in m.named_parameters():v.copy_(torch.tensor(p[k]))
    return m
for path in sorted(p for p in out.glob('*.json') if p.name.startswith(('tied_','untied_'))):
    row=json.loads(path.read_text());source=root/row['input_file'];assert digest(source)==row['input_sha256'];d=np.load(source);saved=np.load(path.with_suffix('.npz'));states=saved['states'];rawstates=saved['rawstates']
    initial={k[8:]:d[k] for k in d.files if k.startswith('initial_')};tp={k[8:]:d[k] for k in d.files if k.startswith('teacher_') and k!='teacher_scale'}
    scale=float(d['student_scale']);tscale=float(d['teacher_scale']);target=evaluate(tp,x)/tscale
    expected=np.concatenate([np.full(a.size,2 if row['student']=='tied_corrected' and k in ['A','B'] else 1) for k,a in initial.items()]);np.testing.assert_array_equal(saved['metric'],expected)
    np.testing.assert_array_equal(rawstates[0],np.concatenate([a.ravel() for a in initial.values()]))
    assert len(states)==len(row['curves'])==len(row['steps'])+1
    t=coefficients(model(tp));keys=sorted(t);G=gram(keys)
    for index,state in enumerate(states):
        p=unpack(state,initial);c=coefficients(model(p));assert set(c)==set(keys)
        raw=unpack(rawstates[index],initial);rc=coefficients(model(raw))
        np.testing.assert_allclose([c[k] for k in keys],[rc[k] for k in keys],atol=1e-10,rtol=1e-10)
        applies=(row['method']=='every' or (row['method']=='initial' and index==0))
        if applies:
            for a,b in [('A','B'),('AR','BR'),('C','D')]:
                if a not in p:continue
                an,bn=np.linalg.norm(raw[a],axis=1),np.linalg.norm(raw[b],axis=1)
                valid=(an>0)&(bn>0)
                np.testing.assert_allclose(np.linalg.norm(p[a],axis=1)[valid],np.linalg.norm(p[b],axis=1)[valid],atol=1e-12,rtol=1e-12)
                # Verify each paired outer product, independently of the rebalancer.
                np.testing.assert_allclose(np.einsum('ij,ik->ijk',p[a],p[b]),np.einsum('ij,ik->ijk',raw[a],raw[b]),atol=1e-11,rtol=1e-11)
        else:np.testing.assert_array_equal(state,rawstates[index])
        delta=np.array([c[k]/scale-t[k]/tscale for k in keys]);mse=float(delta@G@delta)
        error=abs(mse-row['curves'][index]['mse']);assert error<1e-9;worst_moment=max(worst_moment,error)
        if index in {0,len(states)//2,len(states)-1}:
            m=model(p)
            def output(vector):
                parameters={};offset=0
                for k,a in initial.items():parameters[k]=vector[offset:offset+a.size].reshape(a.shape);offset+=a.size
                return torch.func.functional_call(m,parameters,(tx,))/scale
            auto=torch.func.jacrev(output)(torch.tensor(state)).numpy();actual=jacobian(p,x,scale)
            error=float(np.linalg.norm(auto-actual)/max(1,np.linalg.norm(auto)));assert error<1e-10;worst_j=max(worst_j,error)
        if index==len(states)-1:continue
        record=row['steps'][index];J=np.sqrt(w)[:,None]*jacobian(p,x,scale);r=np.sqrt(w)*(evaluate(p,x)/scale-target);g=J.T@r
        direction=(rawstates[index+1]-state)/record['alpha'];residual=(J.T@J+row['damping']*np.diag(expected))@direction+g
        error=float(np.linalg.norm(residual)/max(1,np.linalg.norm(g)));assert error<1e-8;worst_linear=max(worst_linear,error)
        slope=float(g@direction);assert slope<0
        assert abs(slope-record['slope'])<1e-8
        assert record['alpha']==2.**-(record['trials']-1)
        assert .5*row['curves'][index+1]['mse']<=.5*row['curves'][index]['mse']+1e-4*record['alpha']*slope+1e-12
    if row['failure']:
        assert row['failure']['reason']=='line search exhausted'
        p=unpack(states[-1],initial);J=np.sqrt(w)[:,None]*jacobian(p,x,scale)
        r=np.sqrt(w)*(evaluate(p,x)/scale-target);g=J.T@r
        direction=np.linalg.solve(J.T@J+row['damping']*np.diag(expected),-g)
        slope=float(g@direction);assert slope<0
        for alpha in protocol['line_search_alphas']:
            candidate=unpack(states[-1]+alpha*direction,initial)
            rr=np.sqrt(w)*(evaluate(candidate,x)/scale-target)
            assert .5*float(rr@rr)>.5*float(r@r)+protocol['armijo']*alpha*slope
    records.append(dict(name=row['name'],artifact_sha256=digest(path.with_suffix('.npz')),result_sha256=digest(path),failure=row['failure']))
assert len(records)==54
write(out/'audit.json',dict(status='passed',fits=54,records=records,max_relative_jacobian_error=worst_j,max_linear_residual=worst_linear,max_coefficient_moment_error=worst_moment,source_sha256=digest(Path(__file__))))
print('Audited54',worst_j,worst_linear,worst_moment)
