"""Independent NumPy replay of all24 trained toy models and384 deletion evaluations."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;path=P/'validation/capacity_toys.json';r=json.loads(path.read_text());saved=torch.load('/tmp/emlp_t2_models.pt',weights_only=True)
assert len(r['rows'])==len(saved)==24
assert {(x['classes'],x['train_noise'],x['penalty'],x['seed']) for x in r['rows']}=={(c,n,p,s) for c in (4,16) for n in (0.,.25) for p in (0.,.01) for s in (0,1,2)}
forward=[];metric=[];geometry=[]
for s in saved:
 row=r['rows'][s['row']];state={k:v.double().numpy() for k,v in s['state'].items()};x=s['clean_inputs'].double().numpy();y=s['clean_targets'].double().numpy();labels=s['labels'].numpy();centers=s['centers'].double().numpy();h=(x@state['L'].T)*(x@state['R'].T);pred=h@state['D'].T+state['b']
 forward.append(float(np.max(np.abs(pred-s['predictions'].double().numpy()))))
 def check(v,receipt):
  err=float(np.square(v-y).sum(-1).mean());metric.append(abs(err-receipt['squared_error']));guess=(v@centers.T).argmax(-1);C=row['classes'];conf=np.bincount(labels*C+guess,minlength=C*C).reshape(C,C)
  assert np.array_equal(conf,np.array(receipt['confusion']));assert abs(float((guess==labels).mean())-receipt['accuracy'])<1e-7
 check(pred,row['evaluations'][0])
 for mask,receipt in zip(r['masks'],row['deletions']):check((h*np.array(mask))@state['D'].T+state['b'],receipt)
 net=pred-state['b'];energy=(h*h*np.square(state['D']).sum(0)).sum();netenergy=np.square(net).sum();ratio=energy/netenergy;geometry.append(abs(ratio-row['geometry']['channel_net_energy_ratio']))
 assert row['parameters']==544
 if row['classes']==16:assert row['evaluations'][0]['squared_error']>=7/15-1e-6
 # Verify the capacity claim directly from balanced target singular values.
 centered=y-y.mean(0);sv=np.linalg.svd(centered,compute_uv=False);lower=float(np.square(sv[8:]).sum()/len(y))
 assert abs(lower-(7/15 if row['classes']==16 else 0))<1e-6
assert max(forward)<1e-5 and max(metric)<1e-5 and max(geometry)<1e-5
out=dict(status='passed',models=24,clean_and_deletion_evaluations=408,max_forward_error=max(forward),max_metric_error=max(metric),max_energy_ratio_error=max(geometry),capacity_bound_verified=True,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'validation/capacity_toys_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
