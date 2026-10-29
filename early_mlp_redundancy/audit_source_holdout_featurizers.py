"""M59 independent holdout saved-vector and common-target audit."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(x):
 x=np.asarray(x,dtype=np.float64);x=x-x.max();return x-np.log(np.exp(x).sum())
def margin(x,g,v):
 x=np.asarray(x,dtype=np.float64)
 def lse(a):return a.max()+np.log(np.exp(a-a.max()).sum())
 return float(lse(x[g[v]])-lse(x[g[1-v]]))
def main():
 torch.set_num_threads(1);rp=V/'source_holdout_featurizers.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(V/'source_holdout_featurizer_protocol.json');assert r['source_sha256']==sha(P/'probe_source_holdout_featurizers.py');assert r['native_results_sha256']==sha(V/'source_holdout_native.json');assert r['models_sha256']==sha(V/'joint_shared_models.json');n=json.loads((V/'source_holdout_native.json').read_text());sp=Path(n['samples_path']);assert sha(sp)==n['samples_sha256'];native=torch.load(sp,weights_only=True);q=json.loads((V/'source_holdout_native_protocol.json').read_text());nr={z['sample_key']:z for z in n['records'] if z['layer']==1};maxerr=0;checked=0
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(a,b);maxerr=max(maxerr,abs(a-b))
 for f in r['files']:
  path=Path(f['path']);assert sha(path)==f['sha256'];s=torch.load(path,weights_only=True);key=f['key'];assert s['key']==key;nb=native['baselines'][key];p=lp(nb['logits']);g=nb['groups'];v=nb['variant'];pairid,ep=key.rsplit('|',1);pair=next(z for z in q['pairs'] if z['id']==pairid);length=len(pair['token_ids'][int(ep)])
  for b in [b for b in r['baselines'] if b['key']==key]:
   ms=':'.join(str(b[k]) for k in ['kind','seed','arm']);z=s['student_base'][ms];check(float(np.sum(np.exp(p)*(p-lp(z)))),b['final_KL']);assert b['binary']==(margin(z,g,v)>0);assert b['top1']==(int(z.argmax()) in g[v])
  for row in [z for z in r['rows'] if z['key']==key]:
   target=nr[row['native_sample_key']];assert row['status']==target['status'];assert row['family']==pair['family']
   if row['status']=='ineligible':continue
   ms=':'.join(str(row[k]) for k in ['kind','seed','arm']);z=s['student_base'][ms];out=s['student_changed'][ms+'|'+row['region']];native_out=native['changed'][row['native_sample_key']];response=lp(native_out)-p;err=float(np.sum(np.exp(p)*((lp(out)-lp(z))-response)**2));effect=margin(out,g,v)-margin(z,g,v);check(err,row['response_error']);check((effect-target['margin_change'])**2,row['margin_error']);assert row['target_energy']==target['response_energy'];assert row['margin_target_energy']==target['margin_change']**2
   delta=np.zeros((1,length,1152),dtype=np.float32);delta[:,target['recipient_positions']]=native['deltas'][row['native_sample_key']]['delta'].numpy();assert hashlib.sha256(delta.tobytes()).hexdigest()==row['delta_sha256'];checked+=1
   if row['region']=='identity':assert row['response_error']==0 and row['margin_error']==0
 for s in r['summaries']:
  rr=[x for x in r['rows'] if all(x[k]==s[k] for k in ['kind','seed','arm','region']) and (s['family'] is None or x['family']==s['family'])];valid=[x for x in rr if x['status']=='evaluated'];assert len(rr)==s['assigned'] and len(valid)==s['evaluated']
  for field,src in [('target_energy','target_energy'),('error_energy','response_error'),('margin_target_energy','margin_target_energy'),('margin_error_energy','margin_error')]:check(sum(x[src] for x in valid),s[field])
  for numerator,denominator,metric in [('error_energy','target_energy','relative_response_error'),('margin_error_energy','margin_target_energy','relative_margin_error')]:
   if s[denominator]:check(s[numerator]/s[denominator],s[metric])
   else:assert s[metric] is None
 for m in r['macros']:
  rr=[x for x in r['summaries'] if x['family'] is not None and all(x[k]==m[k] for k in ['kind','seed','arm','region'])];valid=[x for x in rr if x['relative_response_error'] is not None];assert len(valid)==m['families_defined'];assert m['families_undefined']==[x['family'] for x in rr if x['relative_response_error'] is None]
  if valid:check(sum(x['relative_response_error'] for x in valid)/len(valid),m['equal_family_response_error'])
  else:assert m['equal_family_response_error'] is None
 assert checked==1512;assert len(r['rows'])==1512 and len(r['baselines'])==504 and len(r['files'])==28;assert r['forward_count']==1596
 out=dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),response_vectors_checked=checked,ineligible_rows=0,baseline_vectors_checked=504,summaries_checked=len(r['summaries']),macro_checks=len(r['macros']),common_delta_hashes_checked=checked,max_metric_absolute_error=maxerr,scope='Every saved final response/baseline independently scored in NumPy, same native target and delta across models, zero/ineligible rows, all summaries/macros. Producer exactly replays28native baselines and84targets; transformer not independently implemented.')
 op=V/'source_holdout_featurizers_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
