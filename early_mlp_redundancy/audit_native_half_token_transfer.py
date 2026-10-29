"""M45 independent token sharing controls and native-output audit."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ls(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def close(a,b,atol=1e-7):assert np.isclose(a,b,rtol=1e-7,atol=atol),(a,b)
rp=P/'native_half_token_transfer.json';r=json.loads(rp.read_text());pp=P/'native_half_token_protocol.json';assert sha(pp)==r['protocol_sha256'];tp=P/'native_half_token_controls.json';assert sha(tp)==r['training_sha256'];tr=json.loads(tp.read_text());assert json.loads((P/'native_half_token_controls_audit.json').read_text())['result_sha256']==sha(tp);assert sha(P.parent/'evaluate_native_half_token_transfer.py')==r['evaluator_sha256'];states={}
for e in tr['models']:
 p=Path(e['path']);assert sha(p)==e['sha256'];states[e['layer'],e['removed']]=torch.load(p,weights_only=True)
old=json.loads((P/'native_half_compensation.json').read_text());assert r['checkpoint_sha256']==old['checkpoint_sha256'];prior={(x['key'],x['layer'],x['removed']):x for x in old['rows'] if x['factor']==1};expected={x[0] for x in prior};seen=set()
assert r['forward_count']==3648 and len(r['rows'])==3456
input_tokens={}
for pair in json.loads((P/'training_source_context_audit.json').read_text())['rows']:
 for ep in [0,1]:input_tokens[f"task:{pair['id']}:{ep}"]=np.array(pair['token_ids'][ep])
cohort=json.loads((P/'natural_token_context_cohort.json').read_text())
for doc in range(32,64):input_tokens[f'natural:{doc}']=np.array(cohort['token_ids'][doc])
for x in r['rows']:
 tok=input_tokens[x['key']];known=np.isin(tok,states[x['layer'],x['removed']]['token_ids'].numpy());assert len(tok)==x['positions'] and int(known.sum())==x['covered_positions']
 key=x['key'],x['layer'],x['removed'],x['arm'];assert key not in seen;seen.add(key);orig=prior[key[:3]]
 for field in ['domain','split','family']:assert x[field]==orig[field]
 for field in ['net_energy','channel_energy']:close(x[field],orig[field],atol=1e-8)
 if x['domain']=='task':assert x['native_task']==orig['native_task']
 close(x['local_relative_error'],x['local_error_energy']/x['net_energy'])
assert seen==set(itertools.product(expected,[0,1,2],['half_01','half_23'],['rank32','token_mean','token_centered_rank32']))
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);decode_errors=[];metric_errors=[]
for s in samples:
 x=r['rows'][s['row']];st=states[x['layer'],x['removed']];ret=s['retained'].numpy();tokens=s['ids'].numpy();keys=st['token_ids'].numpy();idx=np.minimum(np.searchsorted(keys,tokens),len(keys)-1);known=keys[idx]==tokens;im=st['input_mean'].numpy();tm=st['target_mean'].numpy();tx=np.where(known[...,None],st['token_input_mean'].numpy()[idx],im);ty=np.where(known[...,None],st['token_target_mean'].numpy()[idx],tm)
 if x['arm']=='rank32':pred=(ret.astype(float)-im)@st['U'].numpy()@st['B'].numpy()+tm
 elif x['arm']=='token_mean':pred=ty
 else:pred=(ret.astype(float)-tx)@st['U'].numpy()@st['B'].numpy()+ty
 assert int(known.sum())==x['covered_positions'] and tokens.size==x['positions']
 repaired=ret+pred.astype(np.float32);saved=s['repaired'].numpy();err=np.linalg.norm(repaired.astype(float)-saved)/np.linalg.norm(saved);decode_errors.append(err);assert err<1e-6;net=s['net'].numpy().astype(float);close(np.sum((saved.astype(float)-net)**2),x['local_error_energy'])
 energies=np.sum(net**2,axis=-1);errors=np.sum((saved.astype(float)-net)**2,axis=-1)
 for mask,label in [(known,'covered'),(~known,'uncovered')]:close(np.sum(energies[mask]),x[label+'_net_energy']);close(np.sum(errors[mask]),x[label+'_error_energy'])
 a=s['native'].numpy();b=s['output'].numpy();lp=ls(a);lq=ls(b);kl=np.sum(np.exp(lp)*(lp-lq),axis=-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
 for k,value in vals.items():close(value,x[k]);metric_errors.append(abs(value-x[k]))
 if s['groups'] is not None:
  def lse(v):return np.log(np.exp(v-v.max()).sum())+v.max()
  for field,lg in [('native_task',a),('task',b)]:
   groups=s['groups'];variant=s['variant'];last=lg[0,-1].astype(float);margin=lse(last[groups[variant]])-lse(last[groups[1-variant]]);close(margin,x[field]['margin'],atol=1e-10);assert bool(margin>0)==x[field]['binary'];assert bool(int(last.argmax()) in groups[variant])==x[field]['top1']
for s in r['summaries']:
 rr=[x for x in r['rows'] if all(x[k]==s[k] for k in ['domain','split','layer','arm'])];assert len(rr)==s['n']
 for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:close(np.mean([x[k] for x in rr]),s[k])
 close(sum(x['local_error_energy'] for x in rr)/sum(x['net_energy'] for x in rr),s['relative_local_error'])
# Exact replay of all original rank32 outcomes, not only aggregate averages.
previous=json.loads((P/'native_half_substitution_transfer.json').read_text());source={(x['key'],x['layer'],x['removed']):x for x in previous['rows'] if x['arm']=='rank32'};replayed=0;indexed={(x['key'],x['layer'],x['removed'],x['arm']):x for x in r['rows']}
for x in r['rows']:
 close(x['covered_error_energy']+x['uncovered_error_energy'],x['local_error_energy']);close(x['covered_net_energy']+x['uncovered_net_energy'],x['net_energy'])
 if x['arm']=='rank32':
  orig=source[x['key'],x['layer'],x['removed']]
  assert all(x[k]==value for k,value in orig.items())
  other=indexed[x['key'],x['layer'],x['removed'],'token_centered_rank32'];assert x['uncovered_error_energy']==other['uncovered_error_energy'];replayed+=1
assert replayed==1152
out=dict(status='passed',result_sha256=sha(rp),rows_checked=len(seen),rank32_rows_exactly_replayed=replayed,sampled_numpy_checks=len(samples),max_prediction_relative_error=max(decode_errors),max_metric_error=max(metric_errors),native_source_energies_and_task_outputs_replayed=True,scope='All inventories, cohort metadata, source energies, native task outputs and summaries; independent sampled NumPy retained-only prediction and output metrics. No proof of semantic duplication or arbitrary-perturbation response fidelity.')
(P/'native_half_token_transfer_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
