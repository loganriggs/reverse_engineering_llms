"""Independent full-logit metrics and signed-interference audit for M1."""
import json,hashlib
from pathlib import Path
import torch
import numpy as np
P=Path(__file__).resolve().parent;path=P/'validation/matched_layers.json';r=json.loads(path.read_text());s=torch.load('/tmp/emlp_m1_samples.pt',weights_only=True)
assert r['forward_count']==4000 and len(r['rows'])==3840 and len(r['geometry'])==480 and r['D1_MLP1_replay_max_error']==0
errors=[];gerrors=[]
def arr(x):return x.double().numpy()
def ls(x):
 z=x-x.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
for x in s['samples']:
 row=r['rows'][x['row']];a=arr(x['native']);b=arr(x['edited']);p=ls(a);q=ls(b);ids=x['ids'].numpy();k=(np.exp(p)*(p-q)).sum(-1);idx=ids[:,1:,None]
 mm=dict(mean_KL=k.mean(),last_KL=k[0,-1],max_position_KL=k.max(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean(),observed_next_token_dCE=(-np.take_along_axis(q[:,:-1],idx,-1)+np.take_along_axis(p[:,:-1],idx,-1)).mean())
 errors.extend(abs(float(v)-row[f]) for f,v in mm.items())
for x in s['geometry']:
 row=r['geometry'][x['row']];h=arr(x['products']);d=arr(x['Down']);net=h@d.T;mass=np.abs(h)*np.linalg.norm(d,axis=0);energy=np.square(mass).sum(-1);total=mass.sum(-1)
 mm=dict(sum_channel_norms=total.sum(),sum_net_norms=np.linalg.norm(net,axis=-1).sum(),sum_channel_energy=energy.sum(),sum_net_energy=np.square(net).sum(),sum_squared_channel_mass=np.square(total).sum())
 gerrors.extend(abs(float(v)-row[f])/max(abs(float(v)),1e-12) for f,v in mm.items())
for g in r['geometry_summaries']:
 assert abs(g['energy_ratio']-g['sum_channel_energy']/g['sum_net_energy'])<1e-12
 assert abs(g['signed_pair_interference_over_channel_energy']-(g['sum_net_energy']-g['sum_channel_energy'])/g['sum_channel_energy'])<1e-12
for v in r['summaries']:
 rr=[x for x in r['rows'] if x['layer']==v['layer'] and x['kind']==v['kind'] and (v['family'] is None or x['family']==v['family'])];assert len(rr)==v['n']
 for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_squared_error','mean_write_cosine','mean_write_norm_ratio']:
  assert abs(sum(x[f] for x in rr)/len(rr)-v[f])<1e-12
assert max(errors)<1e-7 and max(gerrors)<1e-5
out=dict(status='passed',sampled_output_sequences=len(s['samples']),sampled_native_geometries=len(s['geometry']),max_metric_error=max(errors),max_geometry_relative_error=max(gerrors),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'validation/matched_layers_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
