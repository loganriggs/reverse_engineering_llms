"""Independent NumPy audit of all fixed T4 inputs and channel interventions."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';path=P/'dropout_toys.json';r=json.loads(path.read_text());saved=torch.load('/tmp/emlp_t4_models.pt',weights_only=True)
assert len(saved)==len(r['rows'])==12 and r['new_runs']==9 and r['reused_runs']==3 and r['reused_control_max_error']==0
assert {(x['width'],x['dropout'],x['seed']) for x in r['rows']}=={(w,d,s) for w in (16,32) for d in (0.,.25) for s in (0,1,2)}
forward=[];metrics=[];geom=[];cases=0
for s in saved:
 row=r['rows'][s['row']];w=row['width'];st={k:v.detach().double().numpy() for k,v in s['state'].items()};assert row['parameters']==66*w+16
 def check(v,y,labels,centers,ref):
  global cases
  guess=(v@centers.T).argmax(-1);conf=np.bincount(labels*16+guess,minlength=256).reshape(16,16);assert np.array_equal(conf,np.array(ref['confusion']))
  metrics.append(abs(float(np.square(v-y).sum(-1).mean())-ref['squared_error']));assert abs(float((guess==labels).mean())-ref['accuracy'])<1e-7;cases+=1
  return np.diag(conf)/conf.sum(1)
 for sample,ref in zip(s['evaluation_samples'],row['evaluations']):
  x=sample['inputs'].double().numpy();y=sample['targets'].double().numpy();labels=sample['labels'].numpy();centers=sample['centers'].double().numpy();h=(x@st['L'].T)*(x@st['R'].T);v=h@st['D'].T+st['b'];forward.append(float(np.max(np.abs(v-sample['predictions'].double().numpy()))));check(v,y,labels,centers,ref)
  if sample['noise']==0:
   for mask,d in zip(r['masks'][str(w)],row['deletions']):
    assert sum(mask)==w*3/4;check((h*np.array(mask))@st['D'].T+st['b'],y,labels,centers,d)
   matrix=[]
   for j,d in enumerate(row['singles']):
    hh=h.copy();hh[:,j]=0;matrix.append(check(hh@st['D'].T+st['b'],y,labels,centers,d))
   a=np.array(matrix);target=row['targeted'];assert abs(float(a.min(0).mean())-target['mean_class_accuracy_under_its_worst_channel'])<1e-7;assert int((a.min(0)<1).sum())==target['classes_with_any_single_deletion_error']
   e=h*h*np.square(st['D']).sum(0);net=h@st['D'].T;energy=e.sum();netenergy=np.square(net).sum();mass=(np.abs(h)*np.linalg.norm(st['D'],axis=0)).sum(-1);pc=np.stack([e[labels==i].mean(0) for i in range(16)])
   values=dict(channel_net_energy_ratio=energy/netenergy,participation=np.square(mass).sum()/energy,class_energy_specificity=pc.max(0).sum()/pc.sum())
   geom.extend(abs(float(v)-row['geometry'][k]) for k,v in values.items())
assert cases==516 and max(forward)<1e-5 and max(metrics)<1e-5 and max(geom)<1e-5
out=dict(status='passed',models=12,evaluations=cases,max_forward_error=max(forward),max_metric_error=max(metrics),max_geometry_error=max(geom),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'dropout_toys_audit.json').write_text(json.dumps(out,indent=2)+'\n');(P/'dropout_toy_weights.json').write_text(json.dumps([dict(row=s['row'],state={k:v.tolist() for k,v in s['state'].items()}) for s in saved])+'\n');print(json.dumps(out,indent=2))
