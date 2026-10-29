"""Frozen d=8 representation audit: finite-data readouts and population span floors."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiment import Network, gram, write_csv
from regression_baselines import feature_maps, design


def orthogonal_basis(matrix):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    return u[:,s>s[0]*1e-10]


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;src=root/'results/e1_d8';out=root/'results/readout_refits'
    out.mkdir(parents=True,exist_ok=False)
    grid=[0.,1e-10,1e-8,1e-6,1e-4,1e-3,1e-2,.1,1.,10.]
    protocol=dict(source_run=str(src.relative_to(root)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['experiment.py','regression_baselines.py']},
                  source_results_sha256=hashlib.sha256((src/'results.csv').read_bytes()).hexdigest(),ridge_grid=grid,
                  selection='minimum validation MSE; first grid index breaks ties; no test selection',
                  phases=['trained','initial','teacher'],readouts=['top','hidden'],solvers=['minnorm','ridge_validation'],
                  population_diagnostic='truth-aware projection onto fixed span; lower bound only, never a trained predictor or selected output',
                  metrics='train/validation/test/exact population error, exact population span floor, Gaussian subspace overlap, ranks',
                  exclusions='no extra feature training, no model selection, no GPU, no literal graph-recovery claim',
                  summaries='all 150 trained states; paired comparisons with initial, teacher and original neural errors')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    source=list(csv.DictReader((src/'results.csv').open()))
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),4)})
    H=gram(keys);L=np.linalg.cholesky(H)
    rows=[];geometry=[];checks=[]
    for kind in ['tied','untied']:
        for seed in range(5):
            with np.load(src/f'{kind}_{seed}_64_tied.npz') as data:
                teacher={k.removeprefix('teacher_'):v.copy() for k,v in data.items() if k.startswith('teacher_')}
            teacher_maps=feature_maps(teacher,8,keys);target=teacher_maps['oracle_top']@teacher['w']
            teacher_bases={k:orthogonal_basis(L.T@v) for k,v in teacher_maps.items()}
            rng=torch.Generator().manual_seed(20000+seed)
            inputs=[torch.randn(n,8,generator=rng).numpy() for n in [1024,1024,4096]]
            phis=[design(x,keys) for x in inputs];ys=[p@target for p in phis]
            evidence={'target':target,'keys':np.array(keys)}
            def process(weights,phase,student,n):
                maps=feature_maps(weights,8,keys)
                left=(inputs[1]@weights['A'].T)*(inputs[1]@weights['B'].T)
                right=(inputs[1]@weights['AR'].T)*(inputs[1]@weights['BR'].T) if 'AR' in weights else left
                direct={'oracle_top':(left@weights['C'].T)*(right@weights['D'].T),
                        'oracle_hidden':np.column_stack([left[:,i]*right[:,j] for i in range(len(weights['A'])) for j in range(len(weights['A'])) if 'AR' in weights or i<=j])}
                for map_name,K in maps.items():
                    name=map_name.removeprefix('oracle_');prefix=f'{student}|{n}|{phase}|{name}'
                    features=[p@K for p in phis]
                    direct_error=float(np.max(np.abs(features[1]-direct[map_name]))/max(1.,np.max(np.abs(direct[map_name]))))
                    assert direct_error<1e-10
                    checks.append(direct_error)
                    W=L.T@K;truth=L.T@target;Q=orthogonal_basis(W)
                    pop_beta=np.linalg.lstsq(W,truth,rcond=1e-10)[0]
                    pop_delta=K@pop_beta-target;floor=float(pop_delta@H@pop_delta)
                    stationarity=float(np.linalg.norm(W.T@(W@pop_beta-truth))/max(1.,np.linalg.norm(W.T@truth)))
                    assert stationarity<1e-8
                    ref=teacher_bases[map_name]
                    overlap=float(np.linalg.norm(Q.T@ref)**2/ref.shape[1])
                    evidence[prefix+'|population_beta']=pop_beta
                    geometry.append(dict(teacher=kind,seed=seed,student=student,n=n,phase=phase,readout=name,
                                         coordinates=K.shape[1],functional_rank=Q.shape[1],teacher_rank=ref.shape[1],
                                         teacher_subspace_coverage=overlap,population_span_floor=floor,
                                         population_projection_stationarity=stationarity))
                    Z=features[0][:n];y=ys[0][:n]
                    U,s,Vh=np.linalg.svd(Z,full_matrices=False);uy=U.T@y
                    betas=[];validation=[]
                    for lam in grid:
                        filt=np.divide(1.,s,out=np.zeros_like(s),where=s>s[0]*1e-12) if lam==0 else s/(s*s+n*lam)
                        beta=Vh.T@(filt*uy);betas.append(beta)
                        validation.append(float(np.mean((features[1]@beta-ys[1])**2)))
                    # Store every validation score so an audit can independently check selection.
                    evidence[prefix+'|validation_grid']=np.array(validation)
                    for method,index in [('minnorm',0),('ridge_validation',int(np.argmin(validation)))]:
                        beta=betas[index];coefficient=K@beta;delta=coefficient-target
                        normal=Z.T@(Z@beta-y)/n+grid[index]*beta
                        residual=float(np.linalg.norm(normal)/max(1.,np.linalg.norm(Z.T@y/n)))
                        assert residual<1e-8
                        mse=float(delta@H@delta);assert mse>=floor-1e-9
                        key=prefix+'|'+method;evidence[key]=coefficient;evidence[key+'|beta']=beta
                        rows.append(dict(teacher=kind,seed=seed,student=student,n=n,phase=phase,readout=name,solver=method,
                                         coordinates=K.shape[1],functional_rank=Q.shape[1],design_rank=int(np.sum(s>s[0]*1e-12)),
                                         lambda_selected=grid[index],selected_index=index,
                                         train_mse=float(np.mean((Z@beta-y)**2)),validation_mse=validation[index],
                                         test_mse=float(np.mean((features[2]@beta-ys[2])**2)),exact_gaussian_mse=mse,
                                         population_span_floor=floor,normal_equation_residual=residual,evidence_key=key))
            for n in [64,128,256,512,1024]:
                # Teacher control is recorded once per cohort/sample size, not triplicated by student.
                process(teacher,'teacher','oracle',n)
                for student,width,tied in [('tied',8,True),('untied_width',8,False),('untied_budget',5,False)]:
                    with np.load(src/f'{kind}_{seed}_{n}_{student}.npz') as data:
                        weights={k.removeprefix('student_'):v.copy() for k,v in data.items() if k.startswith('student_')}
                    process(weights,'trained',student,n)
                    initial=Network(8,width,8,tied,30000+seed)
                    if student=='untied_width':
                        with torch.no_grad():initial.AR.copy_(initial.A);initial.BR.copy_(initial.B)
                    weights={k:v.detach().numpy().copy() for k,v in initial.named_parameters()}
                    process(weights,'initial',student,n)
            np.savez_compressed(out/f'{kind}_{seed}.npz',**evidence)
            write_csv(out/'results.csv',rows);write_csv(out/'geometry.csv',geometry)
            print('Completed',kind,seed,'records',len(rows),flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',rows=len(rows),geometry_rows=len(geometry),
                                                      max_direct_feature_relative_error=max(checks)),indent=2)+'\n')


if __name__=='__main__':main()
