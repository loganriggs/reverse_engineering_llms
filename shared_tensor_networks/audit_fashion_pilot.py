"""Independent NumPy tree/stack forwards and classification metrics."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import torch
from fashion_data import load


def forward(row,state,x):
    if row['architecture']=='mlp':
        for i in [0,2,4]:
            x=x@state[f'{i}.weight'].T+state[f'{i}.bias']
            if i!=4:x=np.maximum(x,0)
        return x
    def combine(left,right,i):
        z=(left@state[f'A.{i}'].T)*(right@state[f'B.{i}'].T)
        if row['rmsnorm']:z=z/np.sqrt(np.mean(z*z,axis=1,keepdims=True)+1e-6)
        return z
    if row['architecture']=='stack':
        z=x
        for i in range(row['depth']):z=combine(z,z,i)
    else:
        def visit(i):
            level=row['depth']-1-int(math.log2(i+1))
            return combine(x,x,i) if level==0 else combine(visit(2*i+1),visit(2*i+2),i)
        z=visit(0)
    return z@state['W'].T


def main():
    root=Path(__file__).parent;out=root/'results/fashion_pilot';p=json.loads((out/'protocol.json').read_text());r=json.loads((out/'results.json').read_text());assert r['status']=='complete' and len(r['rows'])==19
    arrays=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');images,labels=arrays[:2];ids=np.load(out/'splits.npz');assert not np.intersect1d(ids['training'],ids['validation']).size
    assert len(ids['training'])==2048 and len(ids['validation'])==5000
    raw=images[ids['training']].astype('float32')/255
    np.testing.assert_allclose([raw.mean(),raw.std()],[p['mean'],p['std']],atol=1e-6,rtol=1e-6)
    vx=(images[ids['validation']].astype('float32')/255-np.float32(p['mean']))/np.float32(p['std']);vy=labels[ids['validation']]
    worst=0.;audited=0;failures=[]
    for row in r['rows']:
        if row['failure']:failures.append(dict(name=row['name'],failure=row['failure']));continue
        state={k:v.numpy() for k,v in torch.load(out/f"{row['name']}.pt",weights_only=True).items()};assert sum(v.size for v in state.values())==row['parameters']
        logits=np.concatenate([forward(row,state,vx[i:i+256]) for i in range(0,len(vx),256)])
        saved=np.load(out/f"{row['name']}_logits.npz")['validation'];err=np.max(np.abs(logits-saved))/max(1.,np.max(np.abs(saved)));assert err<1e-4;worst=max(worst,float(err))
        ce=float(np.mean(logsumexp(logits.astype('float64'),axis=1)-logits[np.arange(len(vy)),vy]));accuracy=float(np.mean(logits.argmax(axis=1)==vy));final=row['curves'][-1]
        assert abs(ce-final['validation_ce'])<1e-4*max(1.,ce)
        assert abs(accuracy-final['validation_accuracy'])<1/5000+1e-7
        audited+=1;print('Audited',row['name'],flush=True)
    # Shared/tree width pair initialized at exactly the same functions.
    for depth in [2,3,4]:
        for norm in [0,1]:
            a=out/f'd{depth}_norm{norm}_stack_logits.npz';b=out/f'd{depth}_norm{norm}_tree_logits.npz'
            if a.exists() and b.exists():np.testing.assert_allclose(np.load(a)['initial'],np.load(b)['initial'],rtol=1e-5,atol=1e-5)
    audit=dict(status='passed',audited_fits=audited,retained_failures=failures,max_relative_numpy_logit_error=worst,official_test_evaluated=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
