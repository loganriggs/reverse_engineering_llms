"""Lossless bilinear checkpoints with a bounded, hash-verified readout layout."""
import hashlib,json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_checkpoint(directory,A,B,W,max_bytes=16*1024**2):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    A,B,W=map(np.asarray,(A,B,W))
    if A.ndim!=2 or B.shape!=A.shape or W.ndim!=2 or W.shape[1]!=len(A)*(len(A)+1)//2:raise ValueError('invalid checkpoint shapes')
    if any(v.dtype not in (np.dtype('float32'),np.dtype('float64')) for v in [A,B,W]):raise ValueError('unsupported dtype')
    rows=(max_bytes-4096)//(W.shape[1]*W.dtype.itemsize)
    if rows<1:raise ValueError('file budget smaller than one readout row')
    factor=directory/'factors.npz';np.savez(factor,A=A,B=B)
    if factor.stat().st_size>max_bytes:raise ValueError('factors exceed file budget')
    chunks=[]
    for start in range(0,len(W),rows):
        end=min(start+rows,len(W));p=directory/f'W_{start:06d}.npz';np.savez(p,W=W[start:end])
        assert p.stat().st_size<=max_bytes
        chunks.append(dict(file=p.name,start=start,end=end,sha256=sha(p)))
    meta=dict(factors=dict(file=factor.name,sha256=sha(factor)),W_shape=list(W.shape),W_dtype=str(W.dtype),chunks=chunks,max_bytes=max_bytes)
    (directory/'checkpoint.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def load_checkpoint(directory):
    directory=Path(directory);meta=json.loads((directory/'checkpoint.json').read_text())
    p=directory/meta['factors']['file'];assert sha(p)==meta['factors']['sha256']
    with np.load(p,allow_pickle=False) as z:A=z['A'].copy();B=z['B'].copy()
    W=np.empty(meta['W_shape'],dtype=meta['W_dtype']);position=0
    for c in meta['chunks']:
        assert c['start']==position and c['end']>position
        p=directory/c['file'];assert sha(p)==c['sha256']
        with np.load(p,allow_pickle=False) as z:
            value=z['W'];assert value.shape==(c['end']-position,W.shape[1]) and value.dtype==W.dtype
            W[position:c['end']]=value
        position=c['end']
    assert position==len(W) and B.shape==A.shape and W.shape[1]==len(A)*(len(A)+1)//2
    return A,B,W
