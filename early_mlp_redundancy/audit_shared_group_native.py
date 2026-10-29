"""M49 full inventory/noop and independent sampled set/decode/output audit."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert np.isclose(a,b,rtol=1e-7,atol=1e-7),(a,b)
def ls(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def main():
 torch.set_num_threads(1);rp=P/'shared_group_native.json';r=json.loads(rp.read_text());pp=P/'shared_group_native_protocol.json';q=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];assert sha(P.parent/'probe_shared_group_native.py')==r['source_sha256'];mp=P/'coactivation_groups.json';assert sha(mp)==r['models_sha256']==q['models_sha256'];assert sha(P/'shared_group_decomposition.json')==q['decomposition_sha256'];models=[x for x in json.loads(mp.read_text())['models'] if x['kind']=='shared_fixed'];assign={x['key']:x['group'] for x in q['assignments']}
 for e in models:assert sha(Path(e['path']))==e['sha256']
 for key,g in assign.items():assert int.from_bytes(hashlib.sha256(('M49:'+key).encode()).digest()[:8],'big')%32==g
 old=json.loads((P/'native_half_compensation.json').read_text());assert old['checkpoint_sha256']==r['checkpoint_sha256'];prior={x['key']:x for x in old['rows'] if x['layer']==1 and x['factor']==1};seen=set();indexed={}
 for x in r['rows']:
  key=x['key'],x['seed'],x['arm'],x['component'];assert key not in seen;seen.add(key);indexed[key]=x;assert x['group']==assign[x['key']];orig=prior[x['key']];assert x['domain']==orig['domain'] and x['split']==orig['split'] and x['family']==orig['family'];assert x['native_write_energy']==orig['net_energy'];assert x['atom_positions']<=8*x['gate_active_positions']
  if x['domain']=='task':assert x['native_task']==orig['native_task']
  if x['noop']:assert x['delta_energy']==x['mean_KL']==x['last_KL']==x['observed_next_token_dCE']==0 and x['top1_agreement']==1
 assert seen=={(key,e['seed'],e['arm'],comp) for key in assign for e in models for comp in ['exclusive','shared','union']};assert len(seen)==4608 and r['forward_count']==192+sum(not x['noop'] for x in r['rows'])
 for key in assign:
  for e in models:
   ex,sh,un=[indexed[key,e['seed'],e['arm'],c] for c in ['exclusive','shared','union']];assert ex['atom_positions']+sh['atom_positions']==un['atom_positions'];assert ex['gate_active_positions']==sh['gate_active_positions']==un['gate_active_positions']
 sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);de=[];me=[]
 for s in samples:
  x=r['rows'][s['row']];M=s['membership'].numpy().astype(bool);g=s['gates'].numpy();a=s['a'].numpy().astype(float);D=s['D'].numpy().astype(float);mask=np.zeros_like(a,dtype=bool)
  for t in range(a.shape[1]):
   active=set(np.flatnonzero(g[0,t]));selected=set(np.flatnonzero(M[x['group']])) if x['group'] in active else set();others=set().union(*(set(np.flatnonzero(M[j])) for j in active-{x['group']}));atoms={'exclusive':selected-others,'shared':selected&others,'union':selected}[x['component']];mask[0,t,list(atoms)]=True
  assert np.array_equal(mask,s['mask'].numpy());assert int(mask.sum())==x['atom_positions'];delta=(a*mask)@D.T*float(s['scale']);cached=s['delta'].numpy();err=np.linalg.norm(delta-cached)/max(np.linalg.norm(cached),1);de.append(err);assert err<1e-5;close(np.sum(cached.astype(float)**2),x['delta_energy'])
  a=s['native'].numpy();b=s['output'].numpy();lp=ls(a);lq=ls(b);kl=np.sum(np.exp(lp)*(lp-lq),axis=-1);ids=s['ids'].numpy();values=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
  for f,val in values.items():close(val,x[f]);me.append(abs(val-x[f]))
  if s['groups'] is not None:
   def lse(z):return np.log(np.exp(z-z.max()).sum())+z.max()
   groups=s['groups'];v=s['variant'];z=b[0,-1].astype(float);margin=lse(z[groups[v]])-lse(z[groups[1-v]]);close(margin,x['task']['margin']);assert bool(margin>0)==x['task']['binary'];assert bool(int(z.argmax()) in groups[v])==x['task']['top1']
 for summary in r['summaries']:
  rr=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['seed','arm','domain','split','component'])];assert len(rr)==summary['n'] and sum(x['noop'] for x in rr)==summary['noops']
  for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:close(sum(x[f] for x in rr)/len(rr),summary[f])
  for f in ['delta_energy','native_write_energy','atom_positions','gate_active_positions']:close(sum(x[f] for x in rr),summary[f])
 out=dict(status='passed',result_sha256=sha(rp),rows_checked=len(seen),noops=sum(x['noop'] for x in r['rows']),native_source_energies_exact=True,independent_set_decode_output_samples=len(samples),max_decode_relative_error=max(de),max_metric_error=max(me),scope='Complete frozen assignments, candidate inventories, noops, native-source replays, membership partitions and summaries; independent sampled set/decode/output metrics. Not evidence of semantic units or native channel identity.')
 (P/'shared_group_native_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
