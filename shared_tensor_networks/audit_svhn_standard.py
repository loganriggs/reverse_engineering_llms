"""Histogram preprocessing audit and independent NumPy classifier replay."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from audit_fashion_pilot import forward

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--depth',type=int,choices=[2,3,4],required=True);parser.add_argument('--partial',action='store_true');args=parser.parse_args();root=Path(__file__).parent;out=root/f'results/svhn_standard/depth{args.depth}';protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256'];assert digest(out/'data_manifest.json')==protocol['array_manifest_sha256']
    if not args.partial:assert json.loads((out/'status.json').read_text())['status']=='complete'
    manifest=json.loads((out/'data_manifest.json').read_text())
    for k in ['images','labels']:assert digest(Path(manifest[k]['path']))==manifest[k]['sha256']
    images=np.load(manifest['images']['path'],mmap_mode='r');labels=np.load(manifest['labels']['path'],mmap_mode='r');ids=np.load(out/'splits.npz');expected=np.random.default_rng(200000).permutation(73257);np.testing.assert_array_equal(ids['training'],expected[:68257]);np.testing.assert_array_equal(ids['validation'],expected[68257:])
    hist=np.zeros((3,256),dtype=np.int64)
    for start in range(0,len(ids['training']),512):
        a=images[ids['training'][start:start+512]]
        for c in range(3):hist[c]+=np.bincount(a[:,c].ravel(),minlength=256)
    values=np.arange(256,dtype='float64')/255;mean=hist@values/hist.sum(1);std=np.sqrt(hist@(values*values)/hist.sum(1)-mean*mean);np.testing.assert_allclose(mean,protocol['mean'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(std,protocol['std'],atol=1e-12,rtol=1e-12)
    records=[];names=[f'd{args.depth}_{k}' for k in ['stack','tree','tree_budget']]+(['d0_mlp'] if args.depth==2 else [])
    for name in names:
        if args.partial and not (out/f'{name}_logits.npz').exists():continue
        row=json.loads((out/f'{name}.json').read_text());cp=out/f'{name}_checkpoint.pt';saved=torch.load(cp,weights_only=True);assert saved['curves']==row['curves'];assert len(row['curves'])==30;state={k:v.numpy() for k,v in saved['model'].items()};assert sum(v.size for v in state.values())==row['parameters']
        for split in ['training','validation']:
            selected=ids[split];logits=[]
            for start in range(0,len(selected),512):
                a=images[selected[start:start+512]].astype('float32');a/=255;a-=mean.astype('float32')[None,:,None,None];a/=std.astype('float32')[None,:,None,None];logits.append(forward(row,state,a.reshape(len(a),3072)))
            z=np.concatenate(logits);yy=labels[selected];assert np.isfinite(z).all();ce=float(np.mean(logsumexp(z.astype('float64'),axis=1)-z[np.arange(len(yy)),yy]));acc=float(np.mean(z.argmax(1)==yy));ref=row['curves'][-1]['train' if split=='training' else 'validation'];assert abs(ce-ref['ce'])<1e-4*max(1,ce);assert abs(acc-ref['accuracy'])<=1/len(yy)+1e-7
            if split=='validation':
                reference=np.load(out/f'{name}_logits.npz')['validation'];error=float(np.max(abs(z-reference))/max(1,np.max(abs(reference))));assert error<1e-4
        records.append(dict(name=name,relative_validation_logit_error=error,checkpoint_sha256=digest(cp),result_sha256=digest(out/f'{name}.json')));print('Audited',name,flush=True)
    if (out/f'd{args.depth}_tree_initial.npy').exists():np.testing.assert_allclose(np.load(out/f'd{args.depth}_stack_initial.npy'),np.load(out/f'd{args.depth}_tree_initial.npy'),rtol=1e-5,atol=1e-5)
    result=dict(status='passed',records=records,source_sha256=digest(Path(__file__)),official_test_used=False);(out/('partial_audit.json' if args.partial else 'audit.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
