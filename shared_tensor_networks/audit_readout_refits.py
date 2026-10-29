"""Check finite readouts using direct features, plus population projection geometry."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment import Network
from regression_baselines import feature_maps


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/readout_refits';src=root/'results/e1_d8'
    protocol=json.loads((out/'protocol.json').read_text());grid=protocol['ridge_grid']
    rows=list(csv.DictReader((out/'results.csv').open()));geo=list(csv.DictReader((out/'geometry.csv').open()))
    assert len(rows)==1400 and len(geo)==700
    identities=[(r['teacher'],r['seed'],r['student'],r['n'],r['phase'],r['readout'],r['solver']) for r in rows]
    assert len(set(identities))==1400
    source={(r['teacher'],r['seed'],r['student'],r['n']):r for r in csv.DictReader((src/'results.csv').open())}
    errors=dict(predictions=0.,normal_equations=0.,population_projection=0.,pythagorean=0.,original_replay=0.)
    selections=0
    for kind in ['tied','untied']:
        for seed in range(5):
            ev=np.load(out/f'{kind}_{seed}.npz');keys=ev['keys'];target=ev['target']
            powers=keys[:,None,:]+keys[None,:,:]
            moments=np.array([0. if k%2 else math.factorial(k)/(2**(k//2)*math.factorial(k//2)) for k in range(9)])
            H=np.prod(moments[powers],axis=2);L=np.linalg.cholesky(H)
            rng=torch.Generator().manual_seed(20000+seed)
            xs=[torch.randn(n,8,generator=rng,dtype=torch.float64).numpy() for n in [1024,1024,4096]]
            phis=[np.column_stack([np.prod(x**k,axis=1) for k in keys]) for x in xs];ys=[p@target for p in phis]
            relevant=[g for g in geo if g['teacher']==kind and int(g['seed'])==seed]
            for g in relevant:
                student=g['student'];n=int(g['n']);phase=g['phase'];name=g['readout'];prefix=f'{student}|{n}|{phase}|{name}'
                if phase=='initial':
                    tied=student=='tied';width=5 if student=='untied_budget' else 8
                    model=Network(8,width,8,tied,30000+seed)
                    if student=='untied_width':
                        with torch.no_grad():model.AR.copy_(model.A);model.BR.copy_(model.B)
                    weights={k:v.detach().numpy() for k,v in model.named_parameters()}
                else:
                    file=src/(f'{kind}_{seed}_{n}_{student}.npz' if phase=='trained' else f'{kind}_{seed}_64_tied.npz')
                    label='student_' if phase=='trained' else 'teacher_'
                    with np.load(file) as data:weights={k.removeprefix(label):v.copy() for k,v in data.items() if k.startswith(label)}
                direct=[]
                for x in xs:
                    l=(x@weights['A'].T)*(x@weights['B'].T)
                    r=(x@weights['AR'].T)*(x@weights['BR'].T) if 'AR' in weights else l
                    if name=='top':features=(l@weights['C'].T)*(r@weights['D'].T)
                    else:features=np.column_stack([l[:,i]*r[:,j] for i in range(l.shape[1]) for j in range(r.shape[1]) if 'AR' in weights or i<=j])
                    direct.append(features)
                K=feature_maps(weights,8,[tuple(k) for k in keys])['oracle_'+name]
                W=L.T@K;truth=L.T@target
                # Independent projection via an orthonormal basis, not the producer's least-squares beta.
                u,s,_=np.linalg.svd(W,full_matrices=False);q=u[:,s>s[0]*1e-10]
                orth_residual=truth-q@(q.T@truth);floor=float(orth_residual@orth_residual)
                err=abs(floor-float(g['population_span_floor']));assert err<1e-10;errors['population_projection']=max(errors['population_projection'],err)
                popbeta=ev[prefix+'|population_beta'];star=K@popbeta
                validation_grid=ev[prefix+'|validation_grid']
                for method in ['minnorm','ridge_validation']:
                    row=next(a for a in rows if a['teacher']==kind and int(a['seed'])==seed and a['student']==student and int(a['n'])==n and a['phase']==phase and a['readout']==name and a['solver']==method)
                    key=prefix+'|'+method;beta=ev[key+'|beta'];coeff=ev[key]
                    np.testing.assert_allclose(K@beta,coeff,rtol=1e-10,atol=1e-10)
                    for metric,Z,P,y in zip(['train_mse','validation_mse','test_mse'],[direct[0][:n],direct[1],direct[2]],[phis[0][:n],phis[1],phis[2]],[ys[0][:n],ys[1],ys[2]]):
                        prediction=Z@beta;error=float(np.max(np.abs(prediction-P@coeff))/max(1.,np.max(np.abs(prediction))))
                        assert error<1e-8;errors['predictions']=max(errors['predictions'],error)
                        value=float(np.mean((prediction-y)**2));assert abs(value-float(row[metric]))<1e-8*max(1.,value)
                    Z=direct[0][:n];y=ys[0][:n];lam=float(row['lambda_selected'])
                    normal=Z.T@(Z@beta-y)/n+lam*beta;error=float(np.linalg.norm(normal)/max(1.,np.linalg.norm(Z.T@y/n)))
                    assert error<1e-8;errors['normal_equations']=max(errors['normal_equations'],error)
                    delta=coeff-target;exact=float(delta@H@delta);assert abs(exact-float(row['exact_gaussian_mse']))<1e-9*max(1.,exact)
                    extra=coeff-star;decomposed=floor+float(extra@H@extra);error=abs(exact-decomposed)/max(1.,exact)
                    assert error<1e-9;errors['pythagorean']=max(errors['pythagorean'],error)
                    if method=='ridge_validation':
                        assert int(row['selected_index'])==int(np.argmin(validation_grid))
                        assert lam==grid[int(row['selected_index'])];selections+=1
                if phase=='trained' and name=='top':
                    c=K@weights['w'];delta=c-target;exact=float(delta@H@delta)
                    expected=float(source[kind,str(seed),student,str(n)]['exact_gaussian_mse'])
                    error=abs(exact-expected)/max(1.,exact);assert error<1e-10;errors['original_replay']=max(errors['original_replay'],error)
            print('Audited',kind,seed,flush=True)
    result=dict(status='passed',rows=len(rows),geometry_rows=len(geo),ridge_selection_records_checked=selections,
                errors=errors,limitation='Selection checked against saved complete validation grid; unselected candidate fits not independently rerun.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256((out/'results.csv').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
