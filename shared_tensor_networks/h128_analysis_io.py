"""Checked artifact access shared by evaluator and audit (no model training)."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from chunked_checkpoint import load_checkpoint
from regenerated_coverage import blocks,digest

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def training(root):
 protocol=json.loads((root/'protocol.json').read_text());done=json.loads((root/'complete.json').read_text());rows=json.loads((root/'progress.json').read_text())['completed']
 expected=set(itertools.product(protocol['teachers'],protocol['ns'],protocol['tasks'],[str(v) for v in protocol['starts']]+['teacher']))
 keys=[(r['teacher'],r['n'],r['tasks'],r['initialization']) for r in rows]
 assert len(keys)==len(set(keys)) and set(keys)==expected and done['fits']==len(expected)
 assert done['status']=='trained_not_evaluated' and done['protocol_sha256']==sha(root/'protocol.json')
 for row in rows:
  case=case_path(root,row);assert sha(case/'trained.json')==row['trained_sha256'] and sha(case/'checkpoint'/'checkpoint.json')==row['checkpoint_manifest_sha256']
  record=json.loads((case/'trained.json').read_text());assert (record['n'],record['tasks'],record['steps'])==(row['n'],row['tasks'],protocol['steps'])
 return protocol,rows

def case_path(root,r):return root/str(r['teacher'])/f'n{r["n"]}_t{r["tasks"]}'/r['initialization']
def case_name(r):return f'{r["teacher"]}_n{r["n"]}_t{r["tasks"]}_{r["initialization"]}'
def data(root,seed):
 td=root/str(seed);A,B,W=load_checkpoint(td/'teacher');p=len(A)*(len(A)+1)//2;recipe=json.loads((td/'source_recipe.json').read_text())
 protocol=json.loads((root/'protocol.json').read_text())
 assert A.shape==(protocol['h'],protocol['d']) and W.shape==(p+4,p)
 assert (recipe['seed'],recipe['stream'],recipe['n'],recipe['tasks'],recipe['block_size'],recipe['noise_sd'])==(seed+5000,0,max(protocol['ns']),p,256,.05)
 x=np.empty((recipe['n'],A.shape[1]),dtype=np.float32);y=np.empty((recipe['n'],p),dtype=np.float32)
 generated=list(blocks(A,B,W[p:],recipe['seed'],json.loads((root/'protocol.json').read_text())['adaptation'],stream=1))
 with np.load(td/'adaptation.npz',allow_pickle=False) as z:
  xa=z['x'];ya=z['y']
  for k in ['x','y','noise']:np.testing.assert_array_equal(z[k],np.concatenate([b[k] for b in generated]))
 for count,b in enumerate(blocks(A,B,W[:p],recipe['seed'],recipe['n']),1):
  r=recipe['blocks'][count-1];assert r==dict(index=b['index'],rows=len(b['x']),sha256=digest(b));start=b['index']*256;x[start:start+len(b['x'])]=b['x'];y[start:start+len(b['x'])]=b['y']
 assert count==len(recipe['blocks'])
 hashes={name:sha(td/name) for name in ['teacher/checkpoint.json','source_recipe.json','adaptation.npz']}
 return A,B,W,x,y,xa,ya,hashes
