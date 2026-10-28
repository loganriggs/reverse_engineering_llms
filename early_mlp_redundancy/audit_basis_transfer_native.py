"""M29 native response audit with unchanged M11-basis benchmark."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x=x-x.max();return x-np.log(np.exp(x).sum())
def lse(x):
 x=x.astype(float);m=x.max();return float(m+np.log(np.exp(x-m).sum()))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
rp=P/'basis_transfer_native_fidelity.json';r=json.loads(rp.read_text());pp=P/'basis_transfer_protocol.json';p=json.loads(pp.read_text());bp=P/'m18_basis_benchmark_protocol.json';benchmark=json.loads(bp.read_text());assert sha(bp)==r['benchmark_protocol_sha256'];oldp=P/'context_native_fidelity.json';assert sha(oldp)==r['benchmark_results_sha256'];old=json.loads(oldp.read_text());assert sha(pp)==r['protocol_sha256'];assert sha(P/'basis_transfer_models.json')==r['training_results_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);states={}
for entry in benchmark['initial_models']:assert sha(Path(entry['path']))==entry['sha256'];states[entry['kind'],entry['seed']]=torch.load(entry['path'],weights_only=True)
errors=[];decodererrors=[];ties=0
for s in samples:
 row=r['records'][s['row']];n,nc,q,qc=[s[k].numpy() for k in ['native','native_changed','reconstructed','reconstructed_changed']];lp,lpc,lq,lqc=map(logsoft,[n,nc,q,qc]);g=s['groups'];v=s['variant'];margin=lambda x:lse(x[g[v]])-lse(x[g[1-v]]);ne=margin(n)-margin(nc);qe=margin(q)-margin(qc);weight=np.exp(lp);nr=lpc-lp;error=(lqc-lq)-nr;values=dict(native_margin_effect=ne,reconstructed_margin_effect=qe,margin_squared_error=(qe-ne)**2,native_margin_energy=ne**2,logprob_response_error=float((weight*error**2).sum()),native_logprob_response_energy=float((weight*nr**2).sum()),intact_final_KL=float((weight*(lp-lq)).sum()),perturbed_final_KL=float((np.exp(lpc)*(lpc-lqc)).sum()))
 for k,val in values.items():errors.append(abs(val-row[k]));close(val,row[k])
 initial=states[row['kind'],row['seed']];st={k:v.numpy().astype(float) for k,v in initial['state'].items()};mu=initial['mean'].numpy();scale=float(initial['scale']);assert np.array_equal(mu,s['mean'].numpy()) and scale==float(s['scale']);a=(s['write'].numpy().astype(float)-mu)/scale@st['encoder'].T+st['encoder_bias'];scores=abs(a) if row['kind']=='scalar' else a**2@st['membership'].T;k=32 if row['kind']=='scalar' else 8;gate=np.zeros_like(scores);np.put_along_axis(gate,np.argsort(scores,axis=-1)[...,-k:],1,axis=-1);support=gate if row['kind']=='scalar' else gate@st['membership'];actual=s['support'].numpy();changed=np.any(support!=actual,axis=-1);ordered=np.sort(scores,axis=-1);gap=(ordered[...,-k]-ordered[...,-k-1])/np.maximum(abs(ordered[...,-k]),1e-12);assert np.all(gap[changed]<1e-5);ties+=int(changed.sum());support[changed]=actual[changed];code=a*support;D=st['dictionary']/np.linalg.norm(st['dictionary'],axis=0,keepdims=True);atoms=s['atoms'].numpy();assert np.array_equal(atoms,np.arange(row['group']*4,row['group']*4+4));mask=s['mask'].numpy();expected=np.zeros_like(mask);boundary=s['prefix_length'];expected[:,:boundary]=1 if row['region']=='prefix' else 0;expected[:,boundary:]=1 if row['region']=='task' else 0;assert np.array_equal(mask,expected);delta=(code[...,atoms]@D[:,atoms].T)*scale*mask;saved=s['delta'].numpy();err=float(np.linalg.norm(delta-saved)/max(np.linalg.norm(saved),1e-20));decodererrors.append(err);assert err<1e-5
expected=set()
for spec in benchmark['causal_interventions']:
 for kind in ['scalar','block']:
  for arm in ['initial']+p['arms']:
   for g in spec['groups']:expected.add((spec['pair_id'],spec['endpoint'],kind,spec['seed'],arm,spec['region'],g))
seen=set();directions={}
for x in r['records']:
 key=tuple(x[k] for k in ['pair_id','endpoint','kind','seed','arm','region','group']);assert key not in seen;seen.add(key);dk=tuple(x[k] for k in ['pair_id','endpoint','kind','seed','region','group']);invariants=tuple(x[k] for k in ['noop','delta_energy','native_margin_effect','native_margin_energy','native_logprob_response_energy']);assert dk not in directions or directions[dk]==invariants;directions[dk]=invariants
 if x['noop']:assert all(x[k]==0 for k in ['delta_energy','native_margin_effect','reconstructed_margin_effect','margin_squared_error','logprob_response_error'])
assert seen==expected and len(seen)==3520 and len(directions)==704;assert sum(v[0] for v in directions.values())==r['noop_directions'];assert r['forward_count']==5148-6*r['noop_directions']
for s in r['summaries']:
 rr=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','arm','region']) and (s['quadrant']=='all' or x['lexical_split']+':'+x['template_split']==s['quadrant']) and (s['family'] is None or x['family']==s['family'])];assert len(rr)==s['interventions'];assert sum(not x['noop'] for x in rr)==s['nonzero']
 for saved,key in [('margin_error_sum','margin_squared_error'),('native_margin_energy_sum','native_margin_energy'),('logprob_response_error_sum','logprob_response_error'),('native_logprob_response_energy_sum','native_logprob_response_energy')]:close(sum(x[key] for x in rr),s[saved])
 for key in ['intact_final_KL','perturbed_final_KL']:close(sum(x[key] for x in rr)/len(rr),s[key])
 close(math.sqrt(s['margin_error_sum']/len(rr)),s['margin_RMSE'])
 for num,den,key in [('margin_error_sum','native_margin_energy_sum','relative_margin_error'),('logprob_response_error_sum','native_logprob_response_energy_sum','relative_logprob_response_error')]:
  if s[den]:close(s[num]/s[den],s[key])
  else:assert s[key] is None
oldbaselines={(x['pair_id'],x['endpoint'],x['kind'],x['seed']):x for x in old['baselines'] if x['arm']=='expanded_context'}
for x in r['baselines']:
 if x['arm']=='initial':
  prior=oldbaselines[x['pair_id'],x['endpoint'],x['kind'],x['seed']];close(x['task']['margin'],prior['task']['margin']);close(x['final_KL'],prior['final_KL'])
reference={}
for filename in ['finite_candidate_native_fidelity.json','aggregate_kl_native_fidelity.json']:
 for x in json.loads((P/filename).read_text())['baselines']:reference[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['arm']]=x
for x in r['baselines']:
 y=reference[x['pair_id'],x['endpoint'],x['kind'],x['seed'],x['arm']]
 for key in ['task','native_task']:assert x[key]==y[key]
 close(x['final_KL'],y['final_KL'])
assert p['basis_protocol_sha256']==sha(bp)
assert benchmark['initial_models']==json.loads((P/'response_supervision_protocol.json').read_text())['initial_models']
parent=json.loads((P/'context_training_protocol.json').read_text());assert benchmark['parent_protocol_sha256']==sha(P/'context_training_protocol.json')
for key in parent:
 if key!='initial_models':assert benchmark[key]==parent[key]
out=dict(status='passed',full_sample_metrics=len(samples),initial_decoder_checks=len(samples),near_tie_support_rows=ties,max_metric_absolute_error=max(errors),max_decoder_relative_error=max(decodererrors),unique_records=len(seen),fixed_native_directions=len(directions),all_candidate_intact_baselines_replayed=True,only_basis_protocol_changed=True,initial_matches_M18_expanded=True,summaries=len(r['summaries']),result_sha256=sha(rp),scope='Independent NumPy initial-code/mask/delta and response metrics on samples; every intervention key/noop/within-basis shared-native invariant, forward count and summary. Does not independently reimplement the native transformer.')
(P/'basis_transfer_native_fidelity_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
