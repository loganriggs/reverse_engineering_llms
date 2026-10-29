"""Predeclared optimization-budget extension, replaying the original 600 steps."""
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import Network, coefficients, gram, write_csv
from regression_baselines import feature_maps


def main():
    torch.set_num_threads(1)
    root=Path(__file__).parent;src=root/'results/e1_d8';out=root/'results/longer_training'
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(steps=[600,1800,6000],ntrain=[256,1024],seeds=list(range(5)),teachers=['tied','untied'],
                  students=['tied','untied_width','untied_budget'],optimizer='full-batch Adam',learning_rate=.003,gradient_clip=10.,
                  dtype='float64',device='cpu',threads=1,initialization='same as original matched d8 run',
                  rationale='compare data-limited and overdetermined regimes after fixed-representation floors remain nonzero',
                  replay='require every 600-step parameter to match existing snapshot exactly; no optimizer restart',
                  evaluation='fixed steps only; no validation/test checkpoint selection',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dependencies={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['experiment.py','regression_baselines.py']},
                  source_results_sha256=hashlib.sha256((src/'results.csv').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),4)})
    H=gram(keys);L=np.linalg.cholesky(H);rows=[];curves=[];replays=[]
    for kind in ['tied','untied']:
        for seed in range(5):
            teacher=Network(8,8,8,kind=='tied',10000+seed)
            with np.load(src/f'{kind}_{seed}_256_tied.npz') as data:
                teacher.load_state_dict({k:torch.from_numpy(data['teacher_'+k].copy()) for k in teacher.state_dict()})
            target=np.array([coefficients(teacher)[k] for k in keys]);truth=L.T@target
            rng=torch.Generator().manual_seed(20000+seed)
            x=torch.randn(1024,8,generator=rng);v=torch.randn(1024,8,generator=rng);test=torch.randn(4096,8,generator=rng)
            with torch.no_grad(): y=teacher(x);yv=teacher(v);yt=teacher(test)
            for n in [256,1024]:
                for name,width,tied in [('tied',8,True),('untied_width',8,False),('untied_budget',5,False)]:
                    model=Network(8,width,8,tied,30000+seed)
                    if name=='untied_width':
                        with torch.no_grad():model.AR.copy_(model.A);model.BR.copy_(model.B)
                    opt=torch.optim.Adam(model.parameters(),lr=.003);clipped=0;started=time.monotonic()
                    for step in range(1,6001):
                        opt.zero_grad(set_to_none=True);loss=(model(x[:n])-y[:n]).square().mean()
                        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {kind} {seed} {n} {name} {step}')
                        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.))
                        if not np.isfinite(gn):raise RuntimeError(f'Nonfinite gradient {kind} {seed} {n} {name} {step}')
                        clipped+=gn>10;opt.step()
                        if step%100==0:
                            with torch.no_grad():curves.append(dict(teacher=kind,seed=seed,n=n,student=name,step=step,
                                                                   train_mse=float((model(x[:n])-y[:n]).square().mean()),gradient_norm=gn))
                        if step not in [600,1800,6000]:continue
                        if step==600:
                            with np.load(src/f'{kind}_{seed}_{n}_{name}.npz') as data:
                                maxerr=max(float(np.max(np.abs(p.detach().numpy()-data['student_'+k]))) for k,p in model.named_parameters())
                                assert maxerr==0.,(kind,seed,n,name,maxerr)
                                replays.append(dict(teacher=kind,seed=seed,n=n,student=name,max_parameter_error=maxerr))
                        pc=coefficients(model);delta=np.array([pc[k] for k in keys])-target
                        weights={k:p.detach().numpy().copy() for k,p in model.named_parameters()}
                        maps=feature_maps(weights,8,keys);floors={}
                        for feature,K in maps.items():
                            W=L.T@K;u,s,_=np.linalg.svd(W,full_matrices=False);q=u[:,s>s[0]*1e-10]
                            residual=truth-q@(q.T@truth);floors[feature+'_population_floor']=float(residual@residual)
                        evidence=f'{kind}_{seed}_{n}_{name}_{step}.npz'
                        np.savez_compressed(out/evidence,**{'student_'+k:p for k,p in weights.items()},
                                            **{'teacher_'+k:p.detach().numpy() for k,p in teacher.named_parameters()})
                        with torch.no_grad():
                            row=dict(teacher=kind,seed=seed,n=n,student=name,step=step,
                                     train_mse=float((model(x[:n])-y[:n]).square().mean()),
                                     validation_mse=float((model(v)-yv).square().mean()),test_mse=float((model(test)-yt).square().mean()),
                                     exact_gaussian_mse=float(delta@H@delta),gradient_norm=gn,clipped_steps=clipped,
                                     cumulative_seconds=time.monotonic()-started,evidence=evidence,**floors)
                        rows.append(row)
                    write_csv(out/'results.csv',rows);write_csv(out/'curves.csv',curves)
                    (out/'replays.json').write_text(json.dumps(replays,indent=2)+'\n')
                    print('Completed',kind,seed,n,name,'records',len(rows),flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',rows=len(rows),exact_replays=len(replays)))+'\n')


if __name__=='__main__':main()
