"""Independent M41 local algebra, output metrics, inventory and prior deletion replay."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ls(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def close(a,b,atol=1e-7):assert np.isclose(a,b,rtol=1e-7,atol=atol),(a,b)
rp=P/'native_half_compensation.json';r=json.loads(rp.read_text());pp=P/'native_half_compensation_protocol.json';q=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];assert sha(P.parent/'probe_native_half_compensation.py')==q['source_sha256'];dp=P/'mlp1_group_deletion.json';assert sha(dp)==q['masks_sha256'];d=json.loads(dp.read_text());m=json.loads((P/'matched_layers.json').read_text());assert r['checkpoint_sha256']==m['checkpoint_sha256'];old={(x['id'],x['endpoint'],x['layer'],x['kind']):x for x in m['rows']};replay=[]
assert len(r['rows'])==2304 and r['forward_count']==2496
expected={f'natural:{i}' for i in range(32,64)}
for x in json.loads((P/'training_source_context_audit.json').read_text())['rows']:
 for ep in [0,1]:expected.add(f"task:{x['id']}:{ep}")
seen=set();paired={}
for x in r['rows']:
 key=(x['key'],x['layer'],x['removed'],x['factor']);assert key not in seen;seen.add(key);paired[key]=x
 close(x['local_relative_error'],x['local_error_energy']/x['net_energy']);close(x['expected_uniform_half_compensation_error'],(4608*x['channel_energy']-x['net_energy'])/4607)
 if x['domain']=='task' and x['factor']==1:
  ident,ep=x['key'][5:].rsplit(':',1);prior=old[ident,int(ep),x['layer'],x['removed']]
  for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:close(x[k],prior[k]);replay.append(abs(x[k]-prior[k]))
assert seen==set(itertools.product(expected,[0,1,2],['half_01','half_23'],[1.,2.]))
# Complementary half compensation errors must match locally: 2A-(A+B)=A-B.
for key in expected:
 for layer in [0,1,2]:close(paired[key,layer,'half_01',2.]['local_error_energy'],paired[key,layer,'half_23',2.]['local_error_energy'],atol=1e-5)
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);metricerr=[];decodererr=[]
for s in samples:
 x=r['rows'][s['row']];prod=s['products'].numpy().astype(float);D=s['D'].numpy().astype(float);net=prod@D.T;keep=np.ones(4608);keep[d['removed_sets'][x['removed']]]=0;ret=(prod*keep)@D.T
 for a,b in [(net,s['net'].numpy()),(ret,s['retained'].numpy())]:err=np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12);decodererr.append(err);assert err<1e-5
 close(np.sum(prod**2*np.sum(D**2,axis=0)),x['channel_energy'])
 # Runtime local arithmetic promotes cached float32 writes; independent decode above checked separately.
 net=s['net'].numpy().astype(float);ret=s['retained'].numpy().astype(float);close(np.sum((x['factor']*ret-net)**2),x['local_error_energy'])
 a=s['native'].numpy();b=s['output'].numpy();lp=ls(a);lq=ls(b);kl=np.sum(np.exp(lp)*(lp-lq),axis=-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
 for k,val in vals.items():close(val,x[k]);metricerr.append(abs(val-x[k]))
for s in r['summaries']:
 rr=[x for x in r['rows'] if all(x[k]==s[k] for k in ['domain','split','layer','factor'])];assert len(rr)==s['n']
 for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:close(np.mean([x[k] for x in rr]),s[k])
 close(sum(x['local_error_energy'] for x in rr)/sum(x['net_energy'] for x in rr),s['relative_local_error']);close(sum(x['expected_uniform_half_compensation_error'] for x in rr)/sum(x['net_energy'] for x in rr),s['expected_uniform_half_relative_error'])
out=dict(status='passed',result_sha256=sha(rp),rows_checked=len(seen),prior_task_deletion_rows_replayed=960,prior_metric_max_error=max(replay),samples_checked=len(samples),max_decoder_relative_error=max(decodererr),max_output_metric_error=max(metricerr),complementary_half_identity=True,scope='Full inventory and summaries; independent sampled NumPy decoding/metrics, original task deletion replay, complementary-half algebra. Not an independent transformer implementation or proof of semantic duplication.')
(P/'native_half_compensation_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
