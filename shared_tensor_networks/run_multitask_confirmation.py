"""Fresh task-holdout study; fixed transfer penalties, no held-out validation labels."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import gram, write_csv
from multitask import MultiTask, make_teacher, monomials
from regression_baselines import design


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/multitask_confirmation';out.mkdir(parents=True,exist_ok=False)
    protocol=dict(d=8,h=4,training_tasks=list(range(12)),heldout_tasks=list(range(12,16)),
                  teacher_types=['shared','independent'],teacher_seeds=list(range(90000,90005)),optimizer_seeds=[92000,92001],
                  input_seeds=list(range(91000,91005)),adaptation_seeds=list(range(93000,93005)),
                  ntrain=[256],adaptation_labels=[8,32,128],steps=3000,optimizer='full-batch Adam',lr=.003,clip=10.,
                  primary_comparisons='teacher-block paired means at k32/128, ridge .001; shared vs separate controls; polynomial comparison retained',
                  models={'shared':dict(h=4,parameters=184),'shared_wide':dict(h=10,parameters=820),'separate_width':dict(h=4,parameters=888),'separate_budget':dict(h=1,parameters=204)},
                  baselines=['random initial predicted training-task outputs','minimum-norm quartic training-task outputs','rank10 truncation of polynomial baseline in Gaussian metric','oracle training-task outputs'],
                  transfer_features='predictions for the 12 training tasks, each normalized by exact Gaussian RMS',
                  transfer_solvers={'minnorm':0.,'ridge_fixed':.001},
                  supervision='exactly k adaptation labels for each heldout task; no heldout validation labels or test selection',
                  metrics='training-task fit, exact IID/OOD Gaussian MSE, target-subspace coverage/rank, task-heldout transfer',
                  limitations='predicted task-output span is not hidden-node or execution-graph recovery; no early stopping',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['multitask.py','experiment.py','regression_baselines.py']})
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    keys=monomials(8);H=gram(keys);L=np.linalg.cholesky(H)
    std=np.linspace(.5,1.5,8);scale=np.prod(std[None,:]**np.array(keys),axis=1);Hood=H*scale[:,None]*scale[None,:]
    fits=[];transfers=[];curves=[];checks=[]
    for kind in ['shared','independent']:
        for ts in range(5):
            C,teacher=make_teacher(8,4,16,kind=='shared',90000+ts,keys,H)
            truth=C[:,:12];u,s,_=np.linalg.svd(L.T@truth,full_matrices=False);Q=u[:,s>s[0]*1e-10]
            assert Q.shape[1]==(10 if kind=='shared' else 12)
            rng=torch.Generator().manual_seed(91000+ts)
            x=torch.randn(256,8,generator=rng);v=torch.randn(1024,8,generator=rng)
            phi=design(x.numpy(),keys);pv=design(v.numpy(),keys)
            y=torch.from_numpy(phi@truth);yv=pv@truth
            arng=torch.Generator().manual_seed(93000+ts);adapt=torch.randn(128,8,generator=arng).numpy()
            pa=design(adapt,keys);ya=pa@C[:,12:]
            evidence={'teacher_coefficients':C,'keys':np.array(keys),'training_inputs':x.numpy(),'validation_inputs':v.numpy(),'adaptation_inputs':adapt,
                      **{'teacher_'+k:a for k,a in teacher.items()}}
            def transfer(F,method,n,optseed):
                norm=np.sqrt(np.sum(F*(H@F),axis=0));assert np.all(norm>1e-12)
                F=F/norm[None,:]
                wf=L.T@F;u,s,_=np.linalg.svd(wf,full_matrices=False);q=u[:,s>s[0]*1e-10]
                coverage=float(np.linalg.norm(q.T@Q)**2/Q.shape[1])
                prefix=f'{n}|{optseed}|{method}';evidence[prefix+'|features']=F
                for k in [8,32,128]:
                    Z=pa[:k]@F;Y=ya[:k];U,sv,Vh=np.linalg.svd(Z,full_matrices=False)
                    for solver,lam in [('minnorm',0.),('ridge_fixed',.001)]:
                        filt=np.divide(1.,sv,out=np.zeros_like(sv),where=sv>sv[0]*1e-12) if lam==0 else sv/(sv*sv+k*lam)
                        B=Vh.T@(filt[:,None]*(U.T@Y));pred=F@B;delta=pred-C[:,12:]
                        normal=Z.T@(Z@B-Y)/k+lam*B
                        assert np.linalg.norm(normal)/max(1.,np.linalg.norm(Z.T@Y/k))<1e-8
                        key=prefix+f'|{k}|{solver}';evidence[key]=B
                        iid=np.sum(delta*(H@delta),axis=0);ood=np.sum(delta*(Hood@delta),axis=0)
                        for j in range(4):
                            transfers.append(dict(teacher=kind,teacher_seed=ts,optimizer_seed=optseed,n=n,method=method,k=k,solver=solver,task=j+12,
                                                  population_mse=float(iid[j]),ood_mse=float(ood[j]),adaptation_mse=float(np.mean((Z@B-Y)[:,j]**2)),
                                                  feature_rank=q.shape[1],teacher_subspace_coverage=coverage,evidence_key=key))
            # Oracle teacher-task span: zero new-task error is possible only for the shared teacher.
            transfer(truth,'oracle_task_outputs',0,-1)
            for n in [256]:
                P=phi[:n];Y=y[:n].numpy()
                polynomial=np.linalg.lstsq(P,Y,rcond=1e-12)[0]
                transfer(polynomial,'polynomial',n,-1)
                up,sp,vhp=np.linalg.svd(L.T@polynomial,full_matrices=False)
                lowrank=np.linalg.solve(L.T,(up[:,:10]*sp[:10])@vhp[:10])
                transfer(lowrank,'polynomial_rank10',n,-1)
                for optseed in range(2):
                    initial_reference=MultiTask(8,4,12,True,92000+optseed)
                    for model_name,h,shared in [('shared',4,True),('shared_wide',10,True),('separate_width',4,False),('separate_budget',1,False)]:
                        model=MultiTask(8,h,12,shared,92000+optseed)
                        assert sum(p.numel() for p in model.parameters())==protocol['models'][model_name]['parameters']
                        if h==4:
                            torch.testing.assert_close(model(x),initial_reference(x),atol=1e-12,rtol=1e-12)
                        init=model.coefficient_matrix(keys)
                        transfer(init,model_name+'_initial',n,optseed)
                        optimizer=torch.optim.Adam(model.parameters(),lr=.003);start=time.monotonic();clipped=0
                        for step in range(1,3001):
                            optimizer.zero_grad(set_to_none=True);loss=(model(x[:n])-y[:n]).square().mean()
                            assert torch.isfinite(loss);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));assert np.isfinite(gn)
                            clipped+=gn>10;optimizer.step()
                            if step%300==0:curves.append(dict(teacher=kind,teacher_seed=ts,optimizer_seed=optseed,n=n,model=model_name,step=step,pre_update_loss=float(loss.detach()),gradient_norm=gn))
                        coeff=model.coefficient_matrix(keys);delta=coeff-truth
                        with torch.no_grad():
                            replay=float(np.max(np.abs(model(v).numpy()-pv@coeff)));assert replay<1e-9;checks.append(replay)
                            train=float((model(x[:n])-y[:n]).square().mean())
                        prefix=f'{n}|{optseed}|{model_name}'
                        evidence[prefix+'|coefficients']=coeff
                        for key,p in model.named_parameters():evidence[prefix+'|'+key]=p.detach().numpy().copy()
                        fits.append(dict(teacher=kind,teacher_seed=ts,optimizer_seed=optseed,n=n,model=model_name,parameters=sum(p.numel() for p in model.parameters()),
                                         initial_population_rms=float(np.sqrt(np.mean(np.sum(init*(H@init),axis=0)))),
                                         train_mse=train,validation_mse=float(np.mean((pv@coeff-yv)**2)),population_mse=float(np.mean(np.sum(delta*(H@delta),axis=0))),
                                         ood_mse=float(np.mean(np.sum(delta*(Hood@delta),axis=0))),seconds=time.monotonic()-start,clipped_steps=clipped))
                        transfer(coeff,model_name,n,optseed)
                        write_csv(out/'fits.csv',fits);write_csv(out/'transfer.csv',transfers);write_csv(out/'curves.csv',curves)
                        print('Completed',kind,ts,n,optseed,model_name,'fits',len(fits),flush=True)
            np.savez_compressed(out/f'{kind}_{ts}.npz',**evidence)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',fits=len(fits),transfer_rows=len(transfers),max_coefficient_forward_error=max(checks)),indent=2)+'\n')


if __name__=='__main__':main()
