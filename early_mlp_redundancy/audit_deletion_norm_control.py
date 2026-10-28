"""Independent saved-logit and write-geometry checks for D2."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;path=P/'validation/deletion_norm_control.json';r=json.loads(path.read_text());d1=json.loads((P/'validation/mlp1_group_deletion.json').read_text())
assert r['forward_count']==1280 and len(r['rows'])==960 and r['zero_native_write_positions']==0 and r['quarter0_replay_metric_max_error']==0
expected={(x['id'],x['endpoint'],x['kind']) for x in d1['rows'] if x['kind'].startswith(('quarter','half'))}
assert {(x['id'],x['endpoint'],x['kind']) for x in r['rows']}==expected
samples=torch.load('/tmp/emlp_d2_samples.pt',weights_only=True);errors=[];normerrors=[];geometryerrors=[]
def array(x):return x.detach().double().numpy()
def logsoft(x):
 z=x-x.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
for s in samples:
 row=r['rows'][s['row']];a=array(s['native']);b=array(s['edited']);ids=s['ids'].numpy();p=logsoft(a);q=logsoft(b);k=(np.exp(p)*(p-q)).sum(-1);target=ids[:,1:,None]
 met=dict(mean_KL=k.mean(),last_KL=k[0,-1],max_position_KL=k.max(),observed_next_token_dCE=(-np.take_along_axis(q[:,:-1],target,-1)+np.take_along_axis(p[:,:-1],target,-1)).mean(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
 errors.extend(abs(float(v)-row[f]) for f,v in met.items())
 y=array(s['native_write']);deleted=array(s['deleted_write']);control=array(s['control_write']);yn=np.linalg.norm(y,axis=-1,keepdims=True);dn=np.linalg.norm(deleted,axis=-1,keepdims=True);expectedcontrol=y*dn/yn
 normerrors.append(float(np.max(np.abs(np.linalg.norm(control,axis=-1,keepdims=True)/dn-1))))
 geometryerrors.append(float(np.linalg.norm(control-expectedcontrol)/np.linalg.norm(expectedcontrol)))
 geom=dict(mean_deleted_native_norm_ratio=(dn/yn).mean(),mean_deleted_native_cosine=((y*deleted).sum(-1)/(yn*dn).squeeze(-1)).mean(),relative_deleted_write_squared_error=np.square(deleted-y).sum()/np.square(y).sum(),relative_control_write_squared_error=np.square(control-y).sum()/np.square(y).sum())
 for f,v in geom.items():assert abs(float(v)-row[f])<1e-6
for s in r['summaries']:
 rr=[x for x in r['rows'] if x['kind']==s['kind'] and (s['family'] is None or x['family']==s['family'])]
 assert len(rr)==s['n']
 for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','mean_deleted_native_norm_ratio','mean_deleted_native_cosine','relative_deleted_write_squared_error','relative_control_write_squared_error']:
  assert abs(sum(x[f] for x in rr)/len(rr)-s[f])<1e-12
assert max(errors)<1e-7 and max(normerrors)<1e-6 and max(geometryerrors)<1e-6
out=dict(status='passed',samples=len(samples),max_metric_error=max(errors),max_norm_relative_error=max(normerrors),max_control_reconstruction_relative_error=max(geometryerrors),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'validation/deletion_norm_control_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
