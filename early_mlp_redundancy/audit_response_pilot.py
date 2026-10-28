"""Recompute one pilot arm with independent NumPy KL and audit stored aggregates."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from inspect_model import load_model,encode,trace
P=Path(__file__).resolve().parent
source=P/'validation/native_mlp2_response_pilot.json'
d=json.loads(source.read_text())
assert d['forward_count']==288 and len(d['rows'])==80
assert d['source_sha256']==hashlib.sha256((P/'probe_mlp2_response.py').read_bytes()).hexdigest()
for kind in ['context','signed_permutation']:
 rows=[r for r in d['rows'] if r['kind']==kind]
 for b in ['live','frozen_write','omitted']:
  assert abs(np.mean([r['metrics'][b]['final_KL'] for r in rows])-d['summary'][kind]['mean_final_KL'][b])<1e-15
 assert all(np.isfinite(r['metrics'][b]['final_KL']) and r['metrics'][b]['final_KL']>=0 for r in rows for b in ['live','frozen_write','omitted'])
torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
m,_=load_model();device=next(m.parameters()).device
ids=encode(d['prompts'][0][0],device);other=encode(d['prompts'][0][1],device)
base,c=trace(m,ids);_,donor=trace(m,other)
delta=donor[1]['native_write'][:,-1]-c[1]['native_write'][:,-1]
def replace(i,z,y):
 if i!=1:return y
 out=y.clone();out[:,-1]+=delta;return out
lg,_=trace(m,ids,replace=replace)
a=base[:,-1].double().cpu().numpy();b=lg[:,-1].double().cpu().numpy()
def logp(x):
 x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
la,lb=logp(a),logp(b);value=float((np.exp(la)*(la-lb)).sum(-1).mean())
expected=d['rows'][0]['metrics']['live']['final_KL'];err=abs(value-expected)
assert err<1e-10
out=dict(status='passed',audited_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 all_aggregate_rows_checked=True,independent_numpy_KL=value,stored_KL=expected,absolute_error=err,
 new_forward_count=3,scope='One live context-edit arm replayed; all stored summary means and finiteness checked. Not a new replication panel.')
(P/'validation/native_mlp2_response_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
