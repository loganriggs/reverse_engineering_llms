"""NumPy M6 scoring and matched-geometry audit on both saved document halves."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(np.float64);z=x-x.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
rp=P/'matched_angle.json';r=json.loads(rp.read_text());sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];assert sha(P/'matched_angle_protocol.json')==r['protocol_sha256'];samples=torch.load(sp,weights_only=True);metric_errors=[];geometry_errors=[];rebuild_errors=[]
for s in samples:
 ref=s['native'].numpy();lp=logsoft(ref);ids=s['ids'].numpy();v=s['v'].numpy().astype(np.float64);t=s['table'].numpy().astype(np.float64);mask=s['mask'].numpy();u=v/np.linalg.norm(v,axis=-1,keepdims=True);parallel=np.sum(t*u,axis=-1,keepdims=True);perp=np.linalg.norm(t-parallel*u,axis=-1,keepdims=True);native_ce=float(-np.take_along_axis(lp[:,:-1],ids[:,1:,None],axis=-1).mean())
 for c in s['conditions']:
  row=next(x for x in r['records'] if x['document']==s['document'] and x['seed']==c['seed']);out=c['output'].numpy();lq=logsoft(out);vals=dict(KL=float((np.exp(lp)*(lp-lq)).sum(-1).mean()),CE=float(-np.take_along_axis(lq[:,:-1],ids[:,1:,None],axis=-1).mean()),top1_agreement=float((ref.argmax(-1)==out.argmax(-1)).mean()),native_CE=native_ce);vals['delta_CE']=vals['CE']-native_ce
  for k,val in vals.items():metric_errors.append(abs(val-row[k]));assert metric_errors[-1]<1e-8
  q=c['q'].numpy().astype(np.float64)
  if c['seed'] is not None:
   g=c['gaussian'].numpy();g=g-np.sum(g*u,axis=-1,keepdims=True)*u;e=g/np.linalg.norm(g,axis=-1,keepdims=True);rebuilt=parallel*u+perp*e;rebuild_errors.append(float(np.linalg.norm(q-rebuilt)/np.linalg.norm(rebuilt)));assert rebuild_errors[-1]<1e-6
   for a,b in [(np.linalg.norm(q,axis=-1),np.linalg.norm(t,axis=-1)),(np.sum(q*u,axis=-1),np.sum(t*u,axis=-1)),(np.sum((q-v)**2,axis=-1),np.sum((t-v)**2,axis=-1))]:
    err=float(np.max(np.abs(a-b)[mask]/np.maximum(np.abs(b[mask]),1e-10)));geometry_errors.append(err);assert err<1e-5
  else:assert np.array_equal(q,t)
  effective=c['effective_write'].numpy();expected=(c['q']+s['bias']).numpy();assert np.array_equal(effective[mask],expected[mask])
for summary in r['summaries']:
 rows=[x for x in r['records'] if x['seed']==summary['seed']];assert len(rows)==64
 for k in ['KL','CE','delta_CE','top1_agreement']:assert abs(float(np.mean([x[k] for x in rows]))-summary[k])<1e-10
 if summary['seed'] is not None:
  diffs=[x['KL']-next(y['KL'] for y in r['records'] if y['document']==x['document'] and y['seed'] is None) for x in rows];assert sum(d>0 for d in diffs)==summary['documents_random_KL_above_table'];assert abs(np.mean(diffs)-summary['mean_paired_KL_difference'])<1e-10
out=dict(status='passed',full_output_audits=8,random_geometry_audits=6,max_metric_absolute_error=max(metric_errors),max_geometry_relative_error=max(geometry_errors),max_reconstruction_relative_error=max(rebuild_errors),all256_record_aggregates_checked=True,result_sha256=sha(rp),scope='Independent NumPy scoring/geometry for documents0/32 and all4arms; production checks geometry on all192random interventions and uncovered writes on all256 interventions. Not independent full-model reimplementation.')
(P/'matched_angle_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
