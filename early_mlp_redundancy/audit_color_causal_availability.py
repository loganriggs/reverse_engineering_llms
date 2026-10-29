"""M56 all saved vectors independently rescored in NumPy."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 z=np.asarray(z,dtype=np.float64);z=z-z.max();return z-np.log(np.exp(z).sum())
def margin(z,g,ep):
 z=np.asarray(z,dtype=np.float64)
 def lse(x):return x.max()+np.log(np.exp(x-x.max()).sum())
 return float(lse(z[g[ep]])-lse(z[g[1-ep]]))
def main():
 torch.set_num_threads(1);rp=V/'color_causal_availability.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(V/'color_causal_availability_protocol.json');assert r['models_sha256']==sha(V/'joint_shared_models.json');assert r['source_sha256']==sha(P/'probe_color_causal_availability.py');sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];s=torch.load(sp,weights_only=True);nr={};maxerr=0
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(a,b);maxerr=max(maxerr,abs(a-b))
 for row in r['native']:
  k=row['key'];c=row['condition'];meta=s['directions'][k+':'+c];delta=meta['delta'].numpy();assert hashlib.sha256(delta.tobytes()).hexdigest()==row['delta_sha256'];assert not np.count_nonzero(delta[:,:-1]);assert np.allclose(s['writes'][k].numpy()[:,-1]+delta[:,-1],meta['donor_final'].numpy(),rtol=1e-6,atol=1e-5);z=s['native'][k+':base'].numpy();y=s['native'][k+':'+c].numpy();g=meta['groups'];ep=meta['endpoint'];p=lp(z);q=lp(y);effect=margin(y,g,ep)-margin(z,g,ep);en=float(np.sum(np.exp(p)*(q-p)**2));check(en,row['native_response_energy']);check(effect,row['native_margin_effect']);check(effect**2,row['native_margin_energy']);check(float(np.sum(np.exp(p)*(p-q))),row['output_KL']);check(float(np.sum(delta.astype(np.float64)**2)),row['delta_energy']);assert (int(z.argmax()) in g[ep])==row['native_top1_before'];assert (int(y.argmax()) in g[ep])==row['native_top1_after'];nr[k,c]=(p,q,effect,en,row)
 for row in r['records']:
  k=row['key'];c=row['condition'];sk=k+'|'+':'.join(str(row[x]) for x in ['kind','seed','arm']);p,q,effect,en,n=nr[k,c];assert row['delta_sha256']==n['delta_sha256'];z=s['student_base'][sk].numpy();y=s['student_changed'][sk+'|'+c].numpy();meta=s['directions'][k+':'+c];seffect=margin(y,meta['groups'],meta['endpoint'])-margin(z,meta['groups'],meta['endpoint']);er=float(np.sum(np.exp(p)*((lp(y)-lp(z))-(q-p))**2));check(er,row['response_error']);check((seffect-effect)**2,row['margin_error']);check(seffect,row['student_margin_effect']);assert row['native_response_energy']==en or np.isclose(row['native_response_energy'],en,rtol=1e-8)
  if c=='identity':assert row['response_error']==0 and row['margin_error']==0
 for row in r['baselines']:
  k=row['key'];sk=k+'|'+':'.join(str(row[x]) for x in ['kind','seed','arm']);p=lp(s['native'][k+':base']);q=lp(s['student_base'][sk]);check(float(np.sum(np.exp(p)*(p-q))),row['final_KL'])
 for summary in r['summaries']:
  rr=[x for x in r['records'] if all(x[k]==summary[k] for k in ['kind','seed','arm','condition'])];assert len(rr)==8
  for field,source in [('target_energy','native_response_energy'),('error_energy','response_error'),('margin_target_energy','native_margin_energy'),('margin_error_energy','margin_error')]:check(sum(x[source] for x in rr),summary[field])
  if summary['target_energy']:check(summary['error_energy']/summary['target_energy'],summary['relative_response_error'])
  else:assert summary['relative_response_error'] is None
 assert len(nr)==24 and len(r['records'])==432 and len(r['baselines'])==144 and len(r['summaries'])==54;assert r['forward_count']==464
 out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),native_vectors_checked=24,student_responses_checked=432,baseline_KLs_checked=144,summaries_checked=54,max_metric_absolute_error=maxerr,common_targets_verified=True,scope='Independent full-vocabulary final-vector NumPy scores and all target deltas/aggregations; no independent transformer implementation. Producer replays all16 native competence vectors exactly.')
 op=V/'color_causal_availability_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
