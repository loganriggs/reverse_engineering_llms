"""M40 audit full output inventory including fixed overlapping unions."""
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
rp=P/'coactivation_outputs.json';r=json.loads(rp.read_text());pp=P/'coactivation_output_protocol.json';protocol=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];split=json.loads((P/'diverse_supervision_split.json').read_text());train=set(split['train_pair_ids']);ev=set(split['evaluation_pair_ids']);models={}
for model in r['models']:
 p=Path(model['path']);assert sha(p)==model['sha256'];models[model['kind'],model['seed'],model['arm']]=torch.load(p,weights_only=True)
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);metricerr=[];forwarderr=[];ties=[]
for s in samples:
 row=r['rows'][s['row']];model=models[row['kind'],row['seed'],row['arm']];st={k:v.numpy() for k,v in model['state'].items()};v=s['write'].numpy().astype(float);mu=model['mean'].numpy();scale=float(model['scale']);x=(v-mu)/scale;a=x@st['encoder'].T+st['encoder_bias'];kind=row['kind'];scores=np.abs(a) if kind=='scalar' else a**2@st['membership'].T;k=32 if kind=='scalar' else (8 if kind=='block' else 4);g=np.zeros_like(scores);np.put_along_axis(g,np.argsort(scores,axis=-1)[...,-k:],1,axis=-1);support=g if kind=='scalar' else (g@st['membership']>0).astype(float);actual=s['support'].numpy();changed=np.any(support!=actual,axis=-1);ordered=np.sort(scores,axis=-1);gap=(ordered[...,-k]-ordered[...,-k-1])/np.maximum(np.abs(ordered[...,-k]),1e-12);assert np.all(gap[changed]<1e-5);ties.append(int(changed.sum()));assert np.all(actual.sum(-1)<=32) and (kind=='shared_fixed' or np.all(actual.sum(-1)==32));support[changed]=actual[changed];D=st['dictionary'].astype(float);D/=np.linalg.norm(D,axis=0,keepdims=True);q=((a*support)@D.T+st['bias'])*scale+mu;cached=s['reconstruction'].numpy();forwarderr.append(float(np.linalg.norm(q-cached)/np.linalg.norm(cached)));assert forwarderr[-1]<1e-5
 ref=s['native'].numpy();out=s['output'].numpy();lp=logsoft(ref);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(ref.argmax(-1)==out.argmax(-1)).mean(),local_relative_error=np.sum((cached.astype(float)-v)**2)/np.sum(v*v))
 for k,val in vals.items():metricerr.append(abs(float(val)-row[k]));assert metricerr[-1]<1e-7
 if s['groups'] is not None:
  for field,lg in [('task',out),('native_task',ref)]:
   last=lg[0,-1].astype(float);groups=s['groups'];variant=s['variant'];margin=lse(last[groups[variant]])-lse(last[groups[1-variant]]);assert abs(margin-row[field]['margin'])<1e-10;assert bool(margin>0)==row[field]['binary'];assert (int(last.argmax()) in groups[variant])==row[field]['top1']
for summary in r['summaries']:
 rows=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['kind','seed','arm','domain','split','template_split']) and (summary['gate']=='all' or x['pass_gate']) and (summary['family'] is None or x['family']==summary['family'])];assert len(rows)==summary['endpoints']
 for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error','active_mean']:assert abs(np.mean([x[k] for x in rows])-summary[k])<1e-10
 if summary['domain']!='natural':
  assert all((x['id'].split('|',1)[-1] if x['domain']=='expanded' else x['id']) in (train if summary['split']=='train' else ev) for x in rows)
  for field,key in [('binary','binary_rate'),('top1','task_top1_rate')]:assert abs(np.mean([x['task'][field] for x in rows])-summary[key])<1e-10
  assert abs(np.mean([abs(x['task']['margin']-x['native_task']['margin']) for x in rows])-summary['absolute_margin_error'])<1e-10
assert len(r['rows'])==13824 and len(r['models'])==12 and r['forward_count']==14976
mp=P/'coactivation_groups.json';assert sha(mp)==r['models_sha256']==protocol['models_sha256'];assert r['models']==json.loads(mp.read_text())['models']
prior=json.loads((P/'joint_shared_outputs.json').read_text())
key=lambda x:tuple(x[k] for k in ['kind','seed','arm','key'])
old={key(x):x for x in prior['rows'] if x['kind']=='scalar' and x['arm']!='initial'}
replayed=0
for row in r['rows']:
 if row['kind']=='scalar':
  assert row==old[key(row)],key(row)
  replayed+=1
assert replayed==4608
# Check complete evaluation inventory, not just aggregates.
expected={f'natural:{i}' for i in range(32,64)}
for row in json.loads((P/'training_source_context_audit.json').read_text())['rows']:
 for ep in [0,1]:expected.add(f"task:{row['id']}:{ep}")
for row in json.loads((P/'context_expansion_protocol.json').read_text())['rows']:
 for ep in [0,1]:expected.add(f"expanded:{row['id']}:{ep}")
for kind,seed,arm in models:
 keys=[x['key'] for x in r['rows'] if (x['kind'],x['seed'],x['arm'])==(kind,seed,arm)];assert len(keys)==len(set(keys))==1152 and set(keys)==expected
out=dict(status='passed',scalar_source_rows_exactly_replayed=replayed,model_hashes_checked=12,independent_forward_and_output_samples=len(samples),near_tie_support_rows=sum(ties),max_forward_relative_error=max(forwarderr),max_metric_absolute_error=max(metricerr),all_records_aggregated=len(r['rows']),result_sha256=sha(rp),scope='Independent NumPy sampled decoder/output metrics and all model hashes, complete inventories and aggregates; not independent native transformer implementation.')
(P/'coactivation_outputs_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
