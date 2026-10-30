import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from chunked_checkpoint import save_checkpoint,load_checkpoint
rng=np.random.default_rng(265301);records=[]
with tempfile.TemporaryDirectory() as tmp:
 for dtype in [np.float32,np.float64]:
    A=rng.normal(size=(128,256)).astype(dtype);B=rng.normal(size=A.shape).astype(dtype);W=rng.normal(size=(257,8256)).astype(dtype)
    root=Path(tmp)/np.dtype(dtype).name;meta=save_checkpoint(root,A,B,W,max_bytes=8*1024**2)
    for x,y in zip([A,B,W],load_checkpoint(root)):np.testing.assert_array_equal(x,y);assert x.dtype==y.dtype
    largest=max(p.stat().st_size for p in root.glob('*.npz'));assert largest<=8*1024**2 and len(meta['chunks'])>1
    reordered=dict(meta,chunks=meta['chunks'][::-1]);(root/'checkpoint.json').write_text(json.dumps(reordered))
    try:load_checkpoint(root)
    except AssertionError:pass
    else:raise AssertionError('reordering accepted')
    (root/'checkpoint.json').write_text(json.dumps(meta));p=root/meta['chunks'][0]['file'];data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    try:load_checkpoint(root)
    except AssertionError:pass
    else:raise AssertionError('corruption accepted')
    records.append(dict(dtype=np.dtype(dtype).name,shape=list(W.shape),chunks=len(meta['chunks']),largest_bytes=largest,exact_roundtrip=True,reordering_rejected=True,corruption_rejected=True))
out=Path('results/chunked_checkpoint_check');out.mkdir(exist_ok=False)
result=dict(records=records,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['chunked_checkpoint.py','check_chunked_checkpoint.py']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
