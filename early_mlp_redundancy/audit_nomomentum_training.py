"""M23 audit paired response objectives, frozen targets and output fidelity."""
import json,hashlib,collections
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def lse(x):
 m=x.max();return float(m+np.log(np.exp(x-m).sum()))
rp=P/'nomomentum_training_results.json';r=json.loads(rp.read_text());pp=P/'nomomentum_protocol.json';protocol=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];split=json.loads((P/'diverse_supervision_split.json').read_text());train=set(split['train_pair_ids']);ev=set(split['evaluation_pair_ids']);models={}
for model in r['models']:
 p=Path(model['path']);assert sha(p)==model['sha256'];models[model['kind'],model['seed'],model['arm']]=torch.load(p,weights_only=True)
for h in r['histories']:
 assert h['optimizer_betas']==[0.,.999] and h['every_first_moment_equal_current_gradient'];assert len(h['updates'])==(0 if h['arm']=='initial' else 120);schedule=protocol['schedules'][str(h['seed'])]
 for step,u in enumerate(h['updates']):
  assert u['step']==step and np.isfinite(u['loss'])
  if step%2==0:expected=[f"natural:{schedule['natural_documents'][step]}"]
  else:
   scheduled=schedule['task_updates'][step//2];assert scheduled['source_pair_id'] in train;prefix='expanded';pair=scheduled['expanded_pair_id'];expected=[f'{prefix}:{pair}:0',f'{prefix}:{pair}:1']
  assert u['keys']==expected
tp=P/'response_training_targets.json';targets=json.loads(tp.read_text());assert sha(tp)==r['targets_sha256'];weights={(x['kind'],x['seed']):x['response_weight'] for x in targets['calibration']}
for h in r['histories']:
 for u in h['updates']:
  components=u['components'];assert len(components)==len(u['keys']);expected_loss=0.
  for ep,c in enumerate(components):
   for k in ['K','R','weight']:assert np.isfinite(c[k])
   if h['arm']=='intact_KL_plus_response' and u['step']%2:
    index=c['target_index'];row=targets['records'][index];assert (row['kind'],row['seed'],row['task_update'],row['endpoint'])==(h['kind'],h['seed'],u['step']//2,ep);assert u['keys'][ep]=='expanded:'+row['input_key'];assert c['weight']==weights[h['kind'],h['seed']]
    if row['noop']:assert c['R']==0
   else:assert c['target_index'] is None and c['weight']==c['R']==0
   expected_loss+=(c['K']+c['weight']*c['R'])/len(components)
  assert np.isclose(u['loss'],expected_loss,rtol=1e-7,atol=1e-9)
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);metricerr=[];forwarderr=[];ties=[]
for s in samples:
 row=r['rows'][s['row']];model=models[row['kind'],row['seed'],row['arm']];st={k:v.numpy() for k,v in model['state'].items()};v=s['write'].numpy().astype(float);mu=model['mean'].numpy();scale=float(model['scale']);x=(v-mu)/scale;a=x@st['encoder'].T+st['encoder_bias'];kind=row['kind'];scores=np.abs(a) if kind=='scalar' else a**2@st['membership'].T;k=32 if kind=='scalar' else 8;g=np.zeros_like(scores);np.put_along_axis(g,np.argsort(scores,axis=-1)[...,-k:],1,axis=-1);support=g if kind=='scalar' else g@st['membership'];actual=s['support'].numpy();changed=np.any(support!=actual,axis=-1);ordered=np.sort(scores,axis=-1);gap=(ordered[...,-k]-ordered[...,-k-1])/np.maximum(np.abs(ordered[...,-k]),1e-12);assert np.all(gap[changed]<1e-5);ties.append(int(changed.sum()));assert np.all(actual.sum(-1)==32);support[changed]=actual[changed];D=st['dictionary'].astype(float);D/=np.linalg.norm(D,axis=0,keepdims=True);q=((a*support)@D.T+st['bias'])*scale+mu;cached=s['reconstruction'].numpy();forwarderr.append(float(np.linalg.norm(q-cached)/np.linalg.norm(cached)));assert forwarderr[-1]<1e-5
 ref=s['native'].numpy();out=s['output'].numpy();lp=logsoft(ref);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(ref.argmax(-1)==out.argmax(-1)).mean(),local_relative_error=np.sum((cached.astype(float)-v)**2)/np.sum(v*v))
 for k,val in vals.items():metricerr.append(abs(float(val)-row[k]));assert metricerr[-1]<1e-7
 if s['groups'] is not None:
  for field,lg in [('task',out),('native_task',ref)]:
   last=lg[0,-1].astype(float);groups=s['groups'];variant=s['variant'];margin=lse(last[groups[variant]])-lse(last[groups[1-variant]]);assert abs(margin-row[field]['margin'])<1e-10;assert bool(margin>0)==row[field]['binary'];assert (int(last.argmax()) in groups[variant])==row[field]['top1']
for summary in r['summaries']:
 rows=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['kind','seed','arm','domain','split','template_split']) and (summary['gate']=='all' or x['pass_gate']) and (summary['family'] is None or x['family']==summary['family'])];assert len(rows)==summary['endpoints']
 for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error']:assert abs(np.mean([x[k] for x in rows])-summary[k])<1e-10
 if summary['domain']!='natural':
  assert all((x['id'].split('|',1)[-1] if x['domain']=='expanded' else x['id']) in (train if summary['split']=='train' else ev) for x in rows)
  for field,key in [('binary','binary_rate'),('top1','task_top1_rate')]:assert abs(np.mean([x['task'][field] for x in rows])-summary[key])<1e-10
  assert abs(np.mean([abs(x['task']['margin']-x['native_task']['margin']) for x in rows])-summary['absolute_margin_error'])<1e-10
assert len(r['rows'])==13824 and len(r['models'])==12 and r['forward_count']==16871
for initial in protocol['initial_models']:
 old=torch.load(initial['path'],weights_only=True);new=models[initial['kind'],initial['seed'],'initial']
 for k,v in old['state'].items():assert torch.equal(v,new['state'][k])
 assert torch.equal(old['mean'],new['mean']) and torch.equal(old['scale'],new['scale'])
# Check complete evaluation inventory, not just aggregates.
expected={f'natural:{i}' for i in range(32,64)}
for row in json.loads((P/'training_source_context_audit.json').read_text())['rows']:
 for ep in [0,1]:expected.add(f"task:{row['id']}:{ep}")
for row in json.loads((P/'context_expansion_protocol.json').read_text())['rows']:
 for ep in [0,1]:expected.add(f"expanded:{row['id']}:{ep}")
for kind,seed,arm in models:
 keys=[x['key'] for x in r['rows'] if (x['kind'],x['seed'],x['arm'])==(kind,seed,arm)];assert len(keys)==len(set(keys))==1152 and set(keys)==expected
out=dict(status='passed',training_schedules_checked=8,frozen_response_components_checked=True,initial_checkpoints_equal=True,model_hashes_checked=12,independent_forward_and_output_samples=len(samples),near_tie_support_rows=sum(ties),max_forward_relative_error=max(forwarderr),max_metric_absolute_error=max(metricerr),all_records_aggregated=len(r['rows']),result_sha256=sha(rp),scope='NumPy featurizer/sample output scores, all paired training schedules and complete evaluation inventories/aggregates; optimizer/backprop/native model not independently reimplemented. Native perturbation validation remains separate and required.')
(P/'nomomentum_training_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
