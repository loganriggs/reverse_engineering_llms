"""Independent native-polynomial verification of the local routing decomposition."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V

def main():
 torch.set_num_threads(1);rp=V/'local_affine_chart_routing.json';r=json.loads(rp.read_text());fit=json.loads((V/'local_affine_chart_fit.json').read_text());anchors={e['name']:np.load(e['path'])['anchors'] for e in fit['models']};reference=json.loads((V/'native_input_reference.json').read_text());assert sha(reference['path'])==reference['sha256'];w=torch.load(reference['path'],weights_only=True);L,R,D=[w[k].double().numpy() for k in ['L','R','D']]
 records={x['sample_key']:x for x in json.loads((V/'whole_task_native.json').read_text())['records']};factor=json.loads((V/'local_affine_chart_factorial.json').read_text());lookup={(x['model'],x['pair_id'],x['region']):[] for x in r['rows']}
 for row in r['rows']:lookup[row['model'],row['pair_id'],row['region']].append(row)
 maximum=0;checked=0
 def chart(x,c):
  dx=x-c;return ((c@L.T)*(c@R.T)+(c@L.T)*(dx@R.T)+(dx@L.T)*(c@R.T))@D.T
 for file in factor['files']:
  assert sha(file['path'])==file['sha256'];b=torch.load(file['path'],weights_only=True);pid=b['pair_id'];z=[x[0].double().numpy() for x in b['inputs']]
  for name,c in anchors.items():
   labels=[np.array([np.argmin(np.sum((c-x)**2,axis=1)) for x in zz]) for zz in z]
   for region in ['changed_input_positions','final_position']:
    rr=lookup.get((name,pid,region),[]);ix=0
    for ep in [0,1]:
     target=records[pid+f'|{ep}|1|'+region]
     if target['status']!='evaluated':continue
     row=rr[ix];ix+=1;pos=target['recipient_positions'];dp=target['donor_positions'];xr=z[ep][pos];xd=z[1-ep][dp];cr=c[labels[ep][pos]];cd=c[labels[1-ep][dp]];qr=chart(xr,cr);qd=chart(xd,cd);qdr=chart(xd,cr);u=qdr-qr;v=qd-qdr;nd=(b['writes'][1-ep][0,dp].double()-b['writes'][ep][0,pos].double()).numpy()
     assert row['switched']==int(np.sum(labels[ep][pos]!=labels[1-ep][dp]))
     metrics=dict(native_energy=np.sum(nd**2),within_energy=np.sum(u**2),routing_energy=np.sum(v**2),cross_term=2*np.sum(u*v),total_energy=np.sum((qd-qr)**2),within_error=np.sum((u-nd)**2),total_error=np.sum((qd-qr-nd)**2))
     for k,value in metrics.items():
      error=abs(float(value)-row[k])/max(1,abs(float(value)));assert error<1e-8,(k,error);maximum=max(maximum,error)
     checked+=1
 for s in r['summaries']:
  rr=[x for x in r['rows'] if x['model']==s['model'] and x['region']==s['region'] and (s['scope']=='all' or x['family']==s['scope'])]
  for k in ['positions','switched','native_energy','within_energy','routing_energy','cross_term','total_energy','within_error','total_error']:assert np.isclose(sum(x[k] for x in rr),s[k],rtol=1e-12,atol=1e-8)
 assert checked==648 and len(r['task_coverage'])==336 and len(r['natural_coverage'])==96
 for row in r['task_coverage']+r['natural_coverage']:assert sum(row['occupancy'])==row['positions']
 result=dict(status='passed',source_sha256=sha(__file__),result_sha256=sha(rp),independent_native_polynomial_cases=checked,max_relative_metric_error=maximum,scope='All task decompositions independently recomputed using factored native polynomial and scalar direct-distance routing; aggregate sums and coverage inventories checked. Natural-distance values are descriptive producer measurements, not independently recomputed here.')
 out=V/'local_affine_chart_routing_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
