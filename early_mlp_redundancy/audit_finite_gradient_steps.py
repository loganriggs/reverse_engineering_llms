"""M25 independent scalar metrics and direction/aggregate accounting."""
from pathlib import Path
import hashlib,json,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
def lp(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
rp=P/'finite_gradient_steps.json';r=json.loads(rp.read_text());pp=P/'finite_gradient_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp)
for key,name in [('targets_sha256','response_training_targets.json'),('training_protocol_sha256','response_supervision_protocol.json'),('gradient_results_sha256','family_gradient_compatibility.json')]:assert p[key]==sha(P/name)
t=json.loads((P/'response_training_targets.json').read_text());sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];metrics=0
for s in samples:
 if 'row' in s:
  x=r['records'][s['row']];a=lp(s['native'].numpy());b=lp(s['output'].numpy());K=float((np.exp(a)*(a-b)).sum(-1).mean());nr=lp(s['native_changed'].numpy())-a[0,-1];fr=lp(s['changed'].numpy())-b[0,-1];R=float((np.exp(a[0,-1])*(fr-nr)**2).sum());close(K,x['K']);close(R,x['R']);errors.extend([abs(K-x['K']),abs(R-x['R'])]);metrics+=1
 else:
  d=next(x for x in r['details'] if (x['kind'],x['seed'])==(s['kind'],s['seed']));g=s['first_gradient'].numpy();raw=s['raw_adam_delta'].numpy();assert np.max(np.abs(raw+.0003*g/(np.abs(g)+1e-8)))<1e-7;close(float(np.linalg.norm(raw)),d['raw_norm']);lookup={x['label']:x['delta'].numpy() for x in s['parameter_deltas']}
  for c in d['conditions']:
   v=lookup[c['label']];close(float(np.linalg.norm(v)),c['post_normalization_norm']);close(float(np.dot(v,s['aggregate_gradient'].numpy())),c['predicted_schedule_change'])
  for name in ['minibatch_adam','aggregate_gradient']:
   for m in [1.,.1]:assert np.array_equal(lookup[f'{name}:{m}:live'],lookup[f'{name}:{m}:fixed'])
training=json.loads((P/'response_supervision_protocol.json').read_text())
models=training['initial_models'];docs={s['natural_documents'][i] for s in training['schedules'].values() for i in range(0,120,2)}
expected=len(docs)+len(models)
for model in models:
 rows=[x for x in t['records'] if (x['kind'],x['seed'])==(model['kind'],model['seed'])];schedule=training['schedules'][str(model['seed'])];natural_count=len({schedule['natural_documents'][i] for i in range(0,120,2)})
 expected+=9*(len({x['input_key'] for x in rows})+sum(not x['noop'] for x in rows)+natural_count)
assert r['forward_count']==expected,(r['forward_count'],expected)
assert metrics==36 and len(r['records'])==4320 and len(r['summaries'])==36
seen=set()
for x in r['records']:
 key=x['target_index'],x['condition'];assert key not in seen;seen.add(key);y=t['records'][x['target_index']];assert all(x[k]==y[k] for k in ['kind','seed','family']);assert 0<=x['live_support_changed_tokens']<=x['tokens']
 if y['noop']:assert x['R']==0
 if x['condition']=='initial:0.0:live':close(x['K'],y['initial_K']);close(x['R'],y['initial_R'])
for s in r['summaries']:
 rows=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','condition'])];assert len(rows)==120
 for k in ['K','R']:close(sum(x[k] for x in rows)/120,s['task_'+k])
 close(sum(x['count']*x['K'] for x in s['natural_records'])/60,s['natural_K']);assert sum(x['count'] for x in s['natural_records'])==60
 close((s['natural_K']+s['task_K']+s['response_weight']*s['task_R'])/2,s['schedule_objective'])
 for fam,vals in s['families'].items():
  rr=[x for x in rows if x['family']==fam]
  for k in vals:close(sum(x[k] for x in rr)/len(rr),vals[k])
 for x in rows:close(x['combined'],x['K']+s['response_weight']*x['R'])
report=[]
for d in r['details']:
 rows=[x for x in r['summaries'] if (x['kind'],x['seed'])==(d['kind'],d['seed'])];base=next(x for x in rows if x['condition']=='initial:0.0:live');lookup={x['condition']:x for x in rows};out=dict(kind=d['kind'],seed=d['seed'],initial_objective=base['schedule_objective'],initial_R=base['task_R'],conditions=[])
 for x in rows:
  if x is base:continue
  fixed=lookup[x['condition'].replace(':live',':fixed')];c=next(c for c in d['conditions'] if c['label']==x['condition']);out['conditions'].append(dict(condition=x['condition'],objective=x['schedule_objective'],objective_change=x['schedule_objective']-base['schedule_objective'],predicted_change=c['predicted_schedule_change'],post_norm=c['post_normalization_norm'],response=x['task_R'],families_response_improved=sum(x['families'][f]['R']<base['families'][f]['R'] for f in x['families']),support_effect=x['schedule_objective']-fixed['schedule_objective']))
 report.append(out);print(json.dumps(out))
out=dict(status='passed',result_sha256=sha(rp),sample_metrics=metrics,records=len(seen),summaries=36,max_metric_error=max(errors),analysis=report,scope='Independent NumPy sampled K/R, actual displacement norms/dots/Adam first-step formula and full scalar aggregation; no independent transformer or support-mask recomputation.')
(P/'finite_gradient_audit.json').write_text(json.dumps(out,indent=2)+'\n')
