"""Independent replay of final full-pool classifiers and their reported metrics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from fashion_data import load
from audit_fashion_pilot import forward

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).parent
    seed_parser=argparse.ArgumentParser(add_help=False);seed_parser.add_argument('--seed',type=int,required=True);seed_args,remaining=seed_parser.parse_known_args();out=root/f'results/fashion_replication/seed{seed_args.seed}'
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args(remaining)
    if not args.partial:assert json.loads((out/'status.json').read_text())['status']=='complete'
    protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
    for n,h in protocol['dependencies'].items():assert digest(root/n)==h
    images,labels,_,_=load('/home/loganriggs/.cache/shared_tensor_networks/fashion-mnist');ids=np.load(out/'splits.npz');expected=np.random.default_rng(150000).permutation(60000)
    np.testing.assert_array_equal(ids['training'],expected[:55000]);np.testing.assert_array_equal(ids['validation'],expected[55000:])
    raw=images[ids['training']].astype('float32')/255
    np.testing.assert_allclose([raw.mean(),raw.std()],[protocol['mean'],protocol['std']],rtol=1e-6,atol=1e-6)
    rows=[]
    for name in [f'd{d}_{k}' for d in [2,3,4] for k in ['stack','tree','tree_budget']]+['d0_mlp']:
        if args.partial and not (out/f'{name}_logits.npz').exists():continue
        row=json.loads((out/f'{name}.json').read_text());cp=out/f'{name}_checkpoint.pt';checkpoint=torch.load(cp,weights_only=True)
        assert row['curves']==checkpoint['curves'];assert [v['epoch'] for v in row['curves']]==list(range(1,31))
        state={k:v.numpy() for k,v in checkpoint['model'].items()};assert sum(a.size for a in state.values())==row['parameters']
        errors={}
        for split in ['training','validation']:
            ix=ids[split];xx=(images[ix].astype('float32')/255-np.float32(protocol['mean']))/np.float32(protocol['std']);yy=labels[ix]
            logits=np.concatenate([forward(row,state,xx[i:i+512]) for i in range(0,len(ix),512)])
            assert np.isfinite(logits).all()
            ce=float(np.mean(logsumexp(logits.astype('float64'),axis=1)-logits[np.arange(len(ix)),yy]));acc=float(np.mean(logits.argmax(1)==yy));ref=row['curves'][-1]['train' if split=='training' else 'validation']
            assert abs(ce-ref['ce'])<1e-4*max(1,ce),(name,split,ce,ref)
            assert abs(acc-ref['accuracy'])<=1/len(ix)+1e-7
            if split=='validation':
                saved=np.load(out/f'{name}_logits.npz')['validation'];err=float(np.max(abs(logits-saved))/max(1,np.max(abs(saved))));assert err<1e-4;errors['relative_logit_error']=err
        rows.append(dict(name=name,checkpoint_sha256=digest(cp),metrics_sha256=digest(out/f'{name}.json'),**errors));print('Audited',name,flush=True)
    result=dict(status='passed',rows=rows,official_test_used=False,source_sha256=digest(Path(__file__)))
    (out/('partial_audit.json' if args.partial else 'audit.json')).write_text(json.dumps(result,indent=2)+'\n');print(f'{len(rows)} final models audited')
if __name__=='__main__':main()
