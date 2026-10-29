"""Deterministic, fixed-block synthetic data with prefix-consistent samples.

Always draw a full block, even for a partial final request. Keep task width fixed
to the stored teacher W, then slice task prefixes at the consumer. Save seed,
stream, block size, NumPy version, teacher arrays and block hashes in experiments.
"""
import hashlib
import numpy as np

def blocks(A,B,W,seed,n,stream=0,block_size=256,noise_sd=.05):
    h,d=A.shape
    assert B.shape==(h,d) and W.shape[1]==h*(h+1)//2
    i,j=np.triu_indices(h)
    for index,start in enumerate(range(0,n,block_size)):
        xrng=np.random.default_rng(np.random.SeedSequence([seed,stream,index,0]))
        nrng=np.random.default_rng(np.random.SeedSequence([seed,stream,index,1]))
        x=xrng.normal(size=(block_size,d));noise=nrng.normal(size=(block_size,len(W)))
        z=(x@A.T)*(x@B.T);y=(z[:,i]*z[:,j])@W.T+noise_sd*noise
        count=min(block_size,n-start)
        yield dict(index=index,x=x[:count],y=y[:count],noise=noise[:count])

def digest(block):
    return {k:hashlib.sha256(np.ascontiguousarray(block[k]).tobytes()).hexdigest() for k in ['x','y','noise']}

def save_readout_chunks(directory,stem,W,rows=512):
    """Lossless dtype-preserving readout chunks, without pickle."""
    from pathlib import Path
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    names=[]
    for index,start in enumerate(range(0,len(W),rows)):
        path=directory/f'{stem}_{index}.npz';np.savez_compressed(path,W=W[start:start+rows]);names.append(path.name)
    return names

def load_readout_chunks(directory,names):
    from pathlib import Path
    arrays=[]
    for name in names:
        with np.load(Path(directory)/name,allow_pickle=False) as saved:arrays.append(saved['W'])
    return np.concatenate(arrays)
