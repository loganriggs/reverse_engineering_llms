"""M79 chart coverage and exact within-chart/routing-change decomposition."""
import json
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import sha,V

def main():
 torch.set_num_threads(1);fit=json.loads((V/'local_affine_chart_fit.json').read_text());states={}
 for e in fit['models']:
  assert sha(e['path'])==e['sha256'];archive=np.load(e['path']);states[e['name']]={k:archive[k] for k in ['anchors','slopes','intercepts']};archive.close()
 def route(x,s):
  d=np.sum((x[:,None,:]-s['anchors'][None,:,:])**2,axis=2);labels=d.argmin(1);return labels,d[np.arange(len(x)),labels]
 def apply(x,labels,s):
  out=np.empty_like(x)
  for k in np.unique(labels):out[labels==k]=x[labels==k]@s['slopes'][k]+s['intercepts'][k]
  return out
 rp=V/'local_affine_chart_factorial.json';result=json.loads(rp.read_text());audit=json.loads((V/'local_affine_chart_factorial_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp)
 native=json.loads((V/'whole_task_native.json').read_text());targets={r['sample_key']:r for r in native['records']};pairs={p['id']:p for p in json.loads((V/'whole_task_native_protocol.json').read_text())['pairs']};rows=[];maxdecomp=0;coverage=[]
 for file in result['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True);pid=b['pair_id'];pair=pairs[pid]
  for name,s in states.items():
   x=[z[0].double().numpy() for z in b['inputs']];routing=[route(z,s) for z in x]
   for ep in (0,1):
    labels,dist=routing[ep];coverage.append(dict(model=name,pair_id=pid,endpoint=ep,positions=len(labels),distance_sum=float(dist.sum()),occupancy=np.bincount(labels,minlength=len(s['anchors'])).tolist()))
    for region in ['changed_input_positions','final_position']:
     target=targets[pid+f'|{ep}|1|'+region]
     if target['status']!='evaluated':continue
     pos=target['recipient_positions'];dp=target['donor_positions'];xr=x[ep][pos];xd=x[1-ep][dp];cr=routing[ep][0][pos];cd=routing[1-ep][0][dp]
     qr=apply(xr,cr,s);qd=apply(xd,cd,s);qdr=apply(xd,cr,s);within=qdr-qr;switch=qd-qdr;total=qd-qr;err=np.linalg.norm(total-within-switch)/max(1,np.linalg.norm(total));assert err<1e-8;maxdecomp=max(maxdecomp,float(err))
     native_delta=(b['writes'][1-ep][0,dp].double()-b['writes'][ep][0,pos].double()).numpy()
     rows.append(dict(model=name,pair_id=pid,family=pair['family'],format=pair['format'],region=region,positions=len(pos),switched=int(np.sum(cr!=cd)),native_energy=float(np.square(native_delta).sum()),within_energy=float(np.square(within).sum()),routing_energy=float(np.square(switch).sum()),cross_term=float(2*np.sum(within*switch)),total_energy=float(np.square(total).sum()),within_error=float(np.square(within-native_delta).sum()),total_error=float(np.square(total-native_delta).sum())))
 ng=[];nr=json.loads((V/'local_affine_chart_natural.json').read_text())
 for file in nr['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True);x=b['input'].double().numpy().reshape(-1,1152)
  for name,s in states.items():
   labels,dist=route(x,s);ng.append(dict(model=name,document=b['document'],positions=len(labels),distance_sum=float(dist.sum()),occupancy=np.bincount(labels,minlength=len(s['anchors'])).tolist()))
 groups=defaultdict(list)
 for r in rows:
  for scope in ['all',r['family']]:groups[r['model'],r['region'],scope].append(r)
 summaries=[]
 for (model,region,scope),rr in groups.items():
  keys=['positions','switched','native_energy','within_energy','routing_energy','cross_term','total_energy','within_error','total_error'];s=dict(model=model,region=region,scope=scope,**{k:sum(r[k] for r in rr) for k in keys});s['switch_fraction']=s['switched']/s['positions'];s['within_relative_error']=s['within_error']/s['native_energy'];s['total_relative_error']=s['total_error']/s['native_energy'];summaries.append(s)
 out=dict(status='completed',source_sha256=sha(__file__),factorial_sha256=sha(rp),natural_sha256=sha(V/'local_affine_chart_natural.json'),rows=rows,summaries=summaries,task_coverage=coverage,natural_coverage=ng,max_decomposition_relative_error=maxdecomp,scope='Descriptive local decomposition only. Within-recipient chart is a counterfactual prediction, not a separately evaluated downstream intervention. Routing norm and error differences do not establish causality or remove amplitude/direction confounding. No chart-switch sample filtering.')
 op=V/'local_affine_chart_routing.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([r for r in summaries if r['scope']=='all'],indent=2))
if __name__=='__main__':main()
