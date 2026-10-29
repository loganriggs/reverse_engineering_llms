"""Audit polynomial coefficients through independent numerical interpolation."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import qr,solve_triangular

from multitask import MultiTask


def direct(A,B,W,x,shared):
    i,j=np.triu_indices(A.shape[-2]);columns=[]
    for t in range(len(W)):
        a=A if shared else A[t];b=B if shared else B[t]
        z=(x@a.T)*(x@b.T);columns.append((z[:,i]*z[:,j])@W[t])
    return np.column_stack(columns)


def main():
    root=Path(__file__).parent;out=root/'results/multitask_transfer'
    fits=list(csv.DictReader((out/'fits.csv').open()));rows=list(csv.DictReader((out/'transfer.csv').open()))
    assert len(fits)==72 and len(rows)==4176
    assert len({(r['teacher'],r['teacher_seed'],r['optimizer_seed'],r['n'],r['model']) for r in fits})==72
    assert len({(r['teacher'],r['teacher_seed'],r['optimizer_seed'],r['n'],r['method'],r['k'],r['solver'],r['task']) for r in rows})==4176
    worst={k:0. for k in ['interpolated_coefficients','fit_metric','transfer_metric','normal_equation','feature_map']};ranks=[]
    for kind in ['shared','independent']:
        for ts in range(3):
            ev=np.load(out/f'{kind}_{ts}.npz');keys=ev['keys'];C=ev['teacher_coefficients']
            power=keys[:,None,:]+keys[None,:,:]
            moment=np.array([0 if k%2 else math.factorial(k)/(2**(k//2)*math.factorial(k//2)) for k in range(9)])
            H=np.prod(moment[power],axis=2);L=np.linalg.cholesky(H)
            std=np.linspace(.5,1.5,8);sc=np.prod(std[None,:]**keys,axis=1);Ho=H*sc[:,None]*sc[None,:]
            def design(x):return np.column_stack([np.prod(x**k,axis=1) for k in keys])
            rng=np.random.default_rng(85000+ts);probe=rng.normal(size=(512,8));P=design(probe)
            Q,R=qr(P,mode='economic');assert np.linalg.matrix_rank(R)==330
            def interpolate(values):return solve_triangular(R,Q.T@values)
            A=ev['teacher_A'];B=ev['teacher_B'];W=ev['teacher_W']
            teacher_probe=direct(A[0] if kind=='shared' else A,B[0] if kind=='shared' else B,W,probe,kind=='shared')
            recovered=interpolate(teacher_probe)
            error=float(np.linalg.norm(recovered-C)/np.linalg.norm(C));assert error<1e-9;worst['interpolated_coefficients']=max(worst['interpolated_coefficients'],error)
            np.testing.assert_allclose(np.sum(C*(H@C),axis=0),np.ones(16),atol=1e-10,rtol=1e-10)
            rank_train=np.linalg.matrix_rank(L.T@C[:,:12],tol=1e-10);rank_all=np.linalg.matrix_rank(L.T@C,tol=1e-10)
            assert (rank_train,rank_all)==((10,10) if kind=='shared' else (12,16))
            ranks.append(dict(teacher=kind,seed=ts,training_rank=int(rank_train),all_task_rank=int(rank_all)))
            xt=ev['training_inputs'];xv=ev['validation_inputs'];xa=ev['adaptation_inputs']
            Pt=design(xt);Pv=design(xv);Pa=design(xa)
            fit_coefs={};feature_coefs={}
            for row in [r for r in fits if r['teacher']==kind and int(r['teacher_seed'])==ts]:
                n=int(row['n']);opt=int(row['optimizer_seed']);model=row['model'];prefix=f'{n}|{opt}|{model}'
                a=ev[prefix+'|A'];b=ev[prefix+'|B'];w=ev[prefix+'|W'];shared=model=='shared'
                predicted=direct(a,b,w,probe,shared);coeff=interpolate(predicted)
                saved=ev[prefix+'|coefficients'];error=float(np.linalg.norm(coeff-saved)/max(1.,np.linalg.norm(saved)))
                assert error<1e-9;worst['interpolated_coefficients']=max(worst['interpolated_coefficients'],error)
                assert a.size+b.size+w.size==int(row['parameters'])
                delta=coeff-C[:,:12]
                metrics={'population_mse':np.mean(np.sum(delta*(H@delta),axis=0)),'ood_mse':np.mean(np.sum(delta*(Ho@delta),axis=0)),
                         'train_mse':np.mean((direct(a,b,w,xt[:n],shared)-Pt[:n]@C[:,:12])**2),
                         'validation_mse':np.mean((direct(a,b,w,xv,shared)-Pv@C[:,:12])**2)}
                for k,value in metrics.items():
                    error=abs(value-float(row[k]))/max(1.,abs(value));assert error<1e-8;worst['fit_metric']=max(worst['fit_metric'],float(error))
                fit_coefs[(n,opt,model)]=coeff
                initial=MultiTask(8,1 if model=='separate_budget' else 4,12,shared,82000+opt)
                ip={k:v.detach().numpy() for k,v in initial.named_parameters()}
                fit_coefs[(n,opt,model+'_initial')]=interpolate(direct(ip['A'],ip['B'],ip['W'],probe,shared))
            for n in [64,256]:
                # Independent QR/pseudoinverse solve for the polynomial baseline.
                poly=np.linalg.lstsq(Pt[:n],Pt[:n]@C[:,:12],rcond=1e-12)[0]
                fit_coefs[(n,-1,'polynomial')]=poly
                u,s,vh=np.linalg.svd(L.T@poly,full_matrices=False)
                fit_coefs[(n,-1,'polynomial_rank10')]=np.linalg.solve(L.T,(u[:,:10]*s[:10])@vh[:10])
            fit_coefs[(0,-1,'oracle_task_outputs')]=C[:,:12]
            for identity,coeff in fit_coefs.items():
                n,opt,method=identity;F=coeff/np.sqrt(np.sum(coeff*(H@coeff),axis=0))[None,:]
                saved=ev[f'{n}|{opt}|{method}|features'];error=float(np.linalg.norm(F-saved)/max(1.,np.linalg.norm(saved)))
                assert error<1e-8;worst['feature_map']=max(worst['feature_map'],error);feature_coefs[identity]=saved
            groups={}
            for row in [r for r in rows if r['teacher']==kind and int(r['teacher_seed'])==ts]:groups.setdefault(row['evidence_key'],[]).append(row)
            for key,group in groups.items():
                assert sorted(int(r['task']) for r in group)==[12,13,14,15]
                row=group[0];n,opt,k=int(row['n']),int(row['optimizer_seed']),int(row['k']);method=row['method']
                F=feature_coefs[n,opt,method];beta=ev[key];Z=Pa[:k]@F;Y=Pa[:k]@C[:,12:]
                lam=0 if row['solver']=='minnorm' else .001
                residual=Z.T@(Z@beta-Y)/k+lam*beta
                error=float(np.linalg.norm(residual)/max(1.,np.linalg.norm(Z.T@Y/k)));assert error<1e-8;worst['normal_equation']=max(worst['normal_equation'],error)
                delta=F@beta-C[:,12:]
                iid=np.sum(delta*(H@delta),axis=0);ood=np.sum(delta*(Ho@delta),axis=0);adapt=np.mean((Z@beta-Y)**2,axis=0)
                for r in group:
                    j=int(r['task'])-12
                    for name,value in [('population_mse',iid[j]),('ood_mse',ood[j]),('adaptation_mse',adapt[j])]:
                        error=abs(float(value)-float(r[name]))/max(1.,abs(value));assert error<1e-8;worst['transfer_metric']=max(worst['transfer_metric'],error)
            print('Audited',kind,ts,flush=True)
    result=dict(status='passed',fits=72,transfer_rows=4176,verification='independent NumPy direct forwards and 512-point QR coefficient interpolation; factorial Gaussian moments',
                max_errors=worst,teacher_ranks=ranks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fit_results_sha256=hashlib.sha256((out/'fits.csv').read_bytes()).hexdigest(),transfer_results_sha256=hashlib.sha256((out/'transfer.csv').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
