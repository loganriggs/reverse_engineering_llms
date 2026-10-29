"""Checksum-verified SVHN cache; no torchvision dependency or test evaluation.
Checksums and pixel/label layout cross-checked with torchvision/datasets/svhn.py.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
from scipy.io import loadmat
SPLITS={'train':(73257,'e26dedcc434d2e4c54c9b2d4a06d8373'),'extra':(531131,'a93ce644f1a588dc4d68dda5feec44a7'),'test':(26032,'eb5a983be6a315427106f1b164d9cef3')}
def load(root,split,base_url="http://ufldl.stanford.edu/housenumbers"):
    n,expected=SPLITS[split];root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/f'{split}_32x32.mat';url=base_url.rstrip('/')+'/'+p.name
    prior_manifest=root/f'{split}_manifest.json'
    if p.exists() and prior_manifest.exists():
        prior=json.loads(prior_manifest.read_text())
        if prior.get('md5')==expected:url=prior['url']
    if not p.exists():
        tmp=p.with_suffix('.download')
        with urllib.request.urlopen(url,timeout=45) as src,tmp.open('wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        with tmp.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()==expected
        tmp.replace(p)
    with p.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()==expected
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    m=loadmat(p);assert m['X'].shape==(32,32,3,n) and m['X'].dtype==np.uint8
    original=m['y'].reshape(-1);assert len(original)==n and set(np.unique(original))==set(range(1,11))
    labels=original.astype('int64')%10;images=m['X'].transpose(3,2,0,1).copy()
    # Pixel layout is N,C,H,W; labels 10 become zero.
    np.testing.assert_array_equal(images[0].transpose(1,2,0),m['X'][:,:,:,0])
    meta=dict(split=split,n=n,url=url,md5=expected,sha256=sha,shape=list(images.shape),label_counts=np.bincount(labels,minlength=10).tolist())
    (root/f'{split}_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
    return images,labels
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--splits',default='train,extra');p.add_argument('--base-url',default='http://ufldl.stanford.edu/housenumbers');args=p.parse_args()
    for split in args.splits.split(','):load('/home/loganriggs/.cache/shared_tensor_networks/svhn',split,args.base_url)
