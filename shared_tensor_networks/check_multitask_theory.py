"""Gradient identities and task-span/rank implications for the frozen cohort."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiment import gram
from multitask import MultiTask


def main():
    torch.set_num_threads(1);torch.manual_seed(87000)
    m=MultiTask(8,4,12,True,87001);u=MultiTask(8,4,12,False,87001)
    x=torch.randn(16,8);target=torch.randn(16,12)
    value=m(x);formula=m(x,recompute=True)
    torch.testing.assert_close(value,formula,rtol=1e-12,atol=1e-12)
    dag_grad=torch.autograd.grad((value-target).square().mean(),tuple(m.parameters()))
    formula_grad=torch.autograd.grad((formula-target).square().mean(),tuple(m.parameters()))
    for a,b in zip(dag_grad,formula_grad):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    occurrence_grad=torch.autograd.grad((u(x)-target).square().mean(),tuple(u.parameters()))
    for i in [0,1]:torch.testing.assert_close(dag_grad[i],occurrence_grad[i].sum(0),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(dag_grad[2],occurrence_grad[2],rtol=1e-12,atol=1e-12)
    root=Path(__file__).parent;out=root/'results/multitask_transfer';records=[]
    for kind in ['shared','independent']:
        for ts in range(3):
            ev=np.load(out/f'{kind}_{ts}.npz');keys=[tuple(k) for k in ev['keys']];H=gram(keys);L=np.linalg.cholesky(H)
            C=ev['teacher_coefficients'];W=L.T@C[:,:12];t=L.T@C[:,12:]
            U,s,_=np.linalg.svd(W,full_matrices=False);r=int(np.sum(s>s[0]*1e-10));Q=U[:,:r]
            residual=t-Q@(Q.T@t);floors=np.sum(residual**2,axis=0)
            if kind=='shared':assert np.max(floors)<1e-20
            else:assert np.min(floors)>1e-6
            bound=float(np.sum(s[10:]**2)/12)
            if kind=='independent':assert bound>1e-6
            records.append(dict(teacher=kind,seed=ts,rank=r,source_rank10_mse_lower_bound=bound,
                                oracle_task_span_holdout_floors=floors.tolist(),retained_smallest_singular_value=float(s[r-1])))
    rows=list(csv.DictReader((out/'transfer.csv').open()))
    perfect=[float(r['population_mse']) for r in rows if r['teacher']=='shared' and r['method']=='oracle_task_outputs' and r['solver']=='minnorm' and int(r['k'])>=32]
    assert len(perfect)==24 and max(perfect)<1e-16
    result=dict(status='passed',dag_formula_gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(dag_grad,formula_grad)),
                parameter_tie_pullback_verified=True,teacher_span_records=records,oracle_exact_transfer_cases=len(perfect),
                maximum_oracle_transfer_error=max(perfect),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'theory_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
