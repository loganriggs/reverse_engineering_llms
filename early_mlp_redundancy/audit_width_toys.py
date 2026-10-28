"""Independent NumPy replay of width16 clean/noisy and deletion metrics."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';path=P/'width_toys.json';r=json.loads(path.read_text());saved=torch.load('/tmp/emlp_t3_models.pt',weights_only=True)
assert len(saved)==len(r['rows'])==12
assert {(x['train_noise'],x['penalty'],x['seed']) for x in r['rows']}=={(n,p,s) for n in (0.,.25) for p in (0.,.01) for s in (0,1,2)}
forward=[];metric=[];geometry=[];bounds=[]
for s in saved:
 row=r['rows'][s['row']];st={k:v.double().numpy() for k,v in s['state'].items()};assert row['parameters']==1072
 def score(v,y,labels,centers,receipt):
  metric.append(abs(float(np.square(v-y).sum(-1).mean())-receipt['squared_error']));guess=(v@centers.T).argmax(-1);conf=np.bincount(labels*16+guess,minlength=256).reshape(16,16)
  assert np.array_equal(conf,np.array(receipt['confusion']));assert abs(float((guess==labels).mean())-receipt['accuracy'])<1e-7
 for sample,receipt in zip(s['evaluation_samples'],row['evaluations']):
  assert sample['noise']==receipt['noise'];x=sample['inputs'].double().numpy();y=sample['targets'].double().numpy();labels=sample['labels'].numpy();centers=sample['centers'].double().numpy();h=(x@st['L'].T)*(x@st['R'].T);v=h@st['D'].T+st['b'];forward.append(float(np.max(np.abs(v-sample['predictions'].double().numpy()))));score(v,y,labels,centers,receipt)
  if sample['noise']==0:
   for mask,d in zip(r['masks'],row['deletions']):
    assert sum(mask)==12;score((h*np.array(mask))@st['D'].T+st['b'],y,labels,centers,d)
   net=h@st['D'].T;energy=(h*h*np.square(st['D']).sum(0)).sum();netenergy=np.square(net).sum();ratio=energy/netenergy;geometry.append(abs(ratio-row['geometry']['channel_net_energy_ratio']));sv=np.linalg.svd(net,compute_uv=False);bound=float(sv.sum()**2/(16*np.square(sv).sum()));assert ratio>=bound-1e-6;bounds.append(dict(row=s['row'],ratio=ratio,bound=bound))
assert max(forward)<1e-5 and max(metric)<1e-5 and max(geometry)<1e-5
out=dict(status='passed',models=12,clean_noisy_and_deletion_evaluations=228,max_forward_error=max(forward),max_metric_error=max(metric),max_energy_ratio_error=max(geometry),geometry_bounds=bounds,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'width_toys_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='geometry_bounds'},indent=2))
(P/'width_toy_weights.json').write_text(json.dumps([dict(row=s['row'],state={k:v.tolist() for k,v in s['state'].items()}) for s in saved])+'\n')
