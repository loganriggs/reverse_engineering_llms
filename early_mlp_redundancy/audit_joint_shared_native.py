"""M37 independent sampled native-effect metrics and complete cross-generator inventory."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(t):
 x=t.numpy().astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def lse(x):
 m=x.max();return m+np.log(np.exp(x-m).sum())
def close(a,b):
 if a is None or b is None:assert a is b;return
 assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
rp=P/'joint_shared_native_fidelity.json';r=json.loads(rp.read_text());mp=P/'joint_shared_models.json';m=json.loads(mp.read_text());assert sha(mp)==r['models_sha256'];bp=P/'context_training_protocol.json';b=json.loads(bp.read_text());assert sha(bp)==r['benchmark_sha256'];oldp=P/'context_native_fidelity.json';assert sha(oldp)==r['benchmark_results_sha256'];old=json.loads(oldp.read_text());assert old['checkpoint_sha256']==r['checkpoint_sha256'];prior={(x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['region'],x['group']):x for x in old['records'] if x['arm']=='initial'}
for e in m['models']:assert sha(Path(e['path']))==e['sha256']
seen=set();directions={};modelkeys={(e['kind'],e['seed'],e['arm']) for e in m['models']}
for x in r['records']:
 mk=x['kind'],x['seed'],x['arm'];assert mk in modelkeys;dk=x['pair_id'],x['endpoint'],x['generator'],x['seed'],x['region'],x['group'];key=mk+dk;assert key not in seen;seen.add(key);p=prior[dk]
 for field in ['native_margin_effect','native_margin_energy','native_logprob_response_energy','delta_energy']:close(x[field],p[field])
 assert x['noop']==p['noop'];close(x['margin_squared_error'],(x['reconstructed_margin_effect']-x['native_margin_effect'])**2)
 if x['noop']:assert x['logprob_response_error']==x['margin_squared_error']==0
 signature=x['delta_sha256'],x['noop'],x['delta_energy']
 if dk in directions:assert directions[dk]==signature
 else:directions[dk]=signature
for mk in modelkeys:
 actual={(x['pair_id'],x['endpoint'],x['generator'],x['seed'],x['region'],x['group']) for x in r['records'] if (x['kind'],x['seed'],x['arm'])==mk}
 assert actual=={k for k in prior if k[3]==mk[1]}
assert len(seen)==6336 and len(directions)==704 and sum(x[1] for x in directions.values())==r['noop_directions']==96
assert len(r['baselines'])==792 and r['forward_count']==44+792+(704-96)*10==6916
assert len({(x['kind'],x['seed'],x['arm'],x['pair_id'],x['endpoint']) for x in r['baselines']})==792
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[]
for s in samples:
 x=r['records'][s['row']];a,b,c,d=[lp(s[k]) for k in ['native','native_changed','reconstructed','reconstructed_changed']];groups=s['groups'];variant=s['variant']
 def margin(t):
  z=t.numpy().astype(float);return lse(z[groups[variant]])-lse(z[groups[1-variant]])
 ne=margin(s['native'])-margin(s['native_changed']);fe=margin(s['reconstructed'])-margin(s['reconstructed_changed']);vals=dict(native_margin_effect=ne,reconstructed_margin_effect=fe,margin_squared_error=(ne-fe)**2,native_margin_energy=ne**2,logprob_response_error=float(np.sum(np.exp(a)*((d-c)-(b-a))**2)),native_logprob_response_energy=float(np.sum(np.exp(a)*(b-a)**2)),intact_final_KL=float(np.sum(np.exp(a)*(a-c))),perturbed_final_KL=float(np.sum(np.exp(b)*(b-d))))
 for k,v in vals.items():close(v,x[k]);errors.append(abs(v-x[k]))
 delta=(s['code'].numpy().astype(float)[...,s['atoms'].numpy()]@s['D'].numpy().astype(float)[:,s['atoms'].numpy()].T)*float(s['scale'])*s['mask'].numpy();saved=s['delta'].numpy();assert np.linalg.norm(delta-saved)<=1e-5*max(np.linalg.norm(saved),1);assert hashlib.sha256(saved.tobytes()).hexdigest()==x['delta_sha256']
for s in r['summaries']:
 rows=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','arm','region']) and (s['generator']=='all' or x['generator']==s['generator']) and (s['quadrant']=='all' or x['lexical_split']+':'+x['template_split']==s['quadrant']) and (s['family'] is None or x['family']==s['family'])];assert len(rows)==s['interventions'] and sum(not x['noop'] for x in rows)==s['nonzero']
 for field,key in [('margin_squared_error','margin_error_sum'),('native_margin_energy','native_margin_energy_sum'),('logprob_response_error','logprob_response_error_sum'),('native_logprob_response_energy','native_logprob_response_energy_sum')]:close(sum(x[field] for x in rows),s[key])
 for num,den,key in [('margin_error_sum','native_margin_energy_sum','relative_margin_error'),('logprob_response_error_sum','native_logprob_response_energy_sum','relative_logprob_response_error')]:close(s[num]/s[den] if s[den] else None,s[key])
out=dict(status='passed',result_sha256=sha(rp),records_checked=6336,native_directions_replayed=704,identical_cross_architecture_deltas=True,sampled_numpy_checks=len(samples),summaries_checked=len(r['summaries']),max_sample_metric_error=max(errors),scope='All old native effects, cross-architecture delta hashes, full benchmark inventories/aggregates; independent sampled NumPy vector effects/decodes. No independent transformer implementation or full optimizer replay.')
(P/'joint_shared_native_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
