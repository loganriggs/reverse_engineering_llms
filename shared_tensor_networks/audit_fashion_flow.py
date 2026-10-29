"""Independent NumPy replay of completed flow-control checkpoints."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from fashion_data import load
from audit_fashion_pilot import forward

def main():
    root=Path(__file__).parent;out=root/'results/fashion_flow';assert json.loads((out/'completed.json').read_text())['cases']==9
    protocol=json.loads((out/'protocol.json').read_text());assert hashlib.sha256((out/'source.py').read_bytes()).hexdigest()==protocol['source_sha256']
    images,labels,_,_=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');ids=np.random.default_rng(150000).permutation(60000);raw=images[ids[:55000]].astype('float64')/255;mean=raw.mean();std=raw.std();del raw
    x=(images[ids[55000:]].astype('float64')/255-mean)/std;y=labels[ids[55000:]];records=[]
    for d in [2,3,4]:
        for seed in protocol['seeds']:
            name=f'd{d}_s{seed}';r=json.loads((out/f'{name}.json').read_text());cp=out/f'{name}.pt';states=torch.load(cp,weights_only=True);logits={}
            assert r['trace'][-1]['step']==215
            for arm,state in states.items():
                row=dict(depth=d,rmsnorm=True,architecture='tree' if arm in ['free','projected'] else 'stack');state={k:v.numpy() for k,v in state.items()};z=np.concatenate([forward(row,state,x[i:i+256]) for i in range(0,len(x),256)]);assert np.isfinite(z).all();logits[arm]=z
                ce=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]));acc=float(np.mean(z.argmax(1)==y));ref=r['trace'][-1]['validation'][arm]
                assert abs(ce-ref['ce'])<1e-9;assert abs(acc-ref['accuracy'])<1e-12
            np.testing.assert_allclose(logits['corrected'],logits['projected'],atol=1e-8,rtol=1e-8)
            p=r['probe'];assert abs(p['free_energy']-p['projected_energy']-p['disagreement_energy'])<1e-9*max(1,p['free_energy'])
            records.append(dict(name=name,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),result_sha256=hashlib.sha256((out/f'{name}.json').read_bytes()).hexdigest()));print('Audited',name,flush=True)
    (out/'audit.json').write_text(json.dumps(dict(status='passed',records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
