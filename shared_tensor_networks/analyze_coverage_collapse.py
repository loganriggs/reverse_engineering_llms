"""Post-result scale diagnostic with cross-fitted scalar corrections."""
import json
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from check_deep_multitask import direct


def main():
    torch.set_num_threads(1);out=Path(__file__).parent/'results/coverage_training'
    rows=json.loads((out/'results.json').read_text())['rows'];answer=[]
    for row in rows:
        depth,seed,method=row['depth'],row['seed'],row['method'];base=f'd{depth}_s{seed}';prefix=f'{base}_{method}'
        teacher=DeepMultiTask(16,8,6,depth,True,0);teacher.load_state_dict(torch.load(out/f'{base}_initial.pt',weights_only=True)['teacher'])
        model=DeepMultiTask(16,8,6,depth,True,0);model.load_state_dict(torch.load(out/f'{prefix}.pt',weights_only=True))
        e=np.load(out/f'{prefix}_evaluation.npz');x=e['inputs'];w=e['weights'];y=direct(teacher,x);pred=direct(model,x)
        n=len(x)//2;corrected=[];alphas=[]
        for train,test in [(slice(0,n),slice(n,None)),(slice(n,None),slice(0,n))]:
            yy,pp,ww=y[train],pred[train],w[train,None]
            denominator=np.sum(ww*pp*pp)
            alpha=float(np.sum(ww*yy*pp)/denominator) if denominator>0 else 0.
            err=np.sum(w[test,None]*(alpha*pred[test]-y[test])**2)/np.sum(w[test,None]*y[test]**2)
            alphas.append(alpha);corrected.append(float(err))
        data=np.load(out/f'{prefix}_training.npz');trainenergy=float(np.mean(data['weights']*np.mean(data['targets']**2,axis=1)))
        q=dict(depth=depth,seed=seed,method=method,training_target_energy=trainenergy,
               prediction_to_target_energy=float(np.sum(w[:,None]*pred**2)/np.sum(w[:,None]*y**2)),
               crossfit_scalars=alphas,crossfit_relative_errors=corrected,original_relative_error=row['relative_mse'])
        answer.append(q);print(depth,seed,method,'energy ratio',q['prediction_to_target_energy'],'rescaled error',np.mean(corrected),flush=True)
    (out/'scale_diagnostic.json').write_text(json.dumps(dict(scope='post-hoc diagnostic; scalar fitted on3 evaluation batches and assessed on other3 then reversed; no feedback to training',rows=answer),indent=2)+'\n')


if __name__=='__main__':main()
