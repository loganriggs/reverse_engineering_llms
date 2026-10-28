"""M28 independently check input exclusions, group decoding, responses and coverage."""
from pathlib import Path
import hashlib,json,collections,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
def lp(x):
 x=x.astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def lse(x):
 m=x.max();return m+np.log(np.exp(x-m).sum())
pp=P/'unseen_direction_protocol.json';p=json.loads(pp.read_text());rp=P/'unseen_direction_results.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp);tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(tp)==p['targets_sha256']==r['targets_sha256'];cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);identity={k:hashlib.sha256(json.dumps(v['ids'].tolist(),separators=(',',':')).encode()).hexdigest() for k,v in cache['inputs'].items()};seen=collections.defaultdict(set)
for x in t['records']:seen[x['kind'],x['seed'],identity[x['input_key']]].add(x['group'])
assert len(p['rows'])==480 and len(p['models'])==20
for i,s in enumerate(p['rows']):
 x=t['records'][i];assert s['target_index']==i and s['token_hash']==identity[x['input_key']];assert all(s[k]==x[k] for k in ['kind','seed','family','region','input_key']);forbidden=seen[x['kind'],x['seed'],s['token_hash']];assert s['supervised_groups']==sorted(forbidden);available=set(range(32))-forbidden;key=lambda g:hashlib.sha256(f"M28-unseen-v1|{x['kind']}|{x['seed']}|{s['token_hash']}|{x['region']}|{g}".encode()).hexdigest();assert s['group']==min(available,key=key)
for e in p['models']:assert sha(Path(e['path']))==e['sha256']
source=json.loads((P/'response_supervision_protocol.json').read_text());states={}
for e in source['initial_models']:
 assert sha(Path(e['path']))==e['sha256'];states[e['kind'],e['seed']]=torch.load(e['path'],weights_only=True)
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];decode_errors=[];ties=0
for sample in samples:
 x=r['records'][sample['row']];i=x['target_index'];spec=p['rows'][i];target=t['records'][i];c=cache['inputs'][target['input_key']];initial=cache['initials'][target['initial_key']];s=states[x['kind'],x['seed']];st={k:v.numpy().astype(float) for k,v in s['state'].items()};mu=s['mean'].numpy();scale=float(s['scale']);a=(c['write'].numpy().astype(float)-mu)/scale@st['encoder'].T+st['encoder_bias'];scores=abs(a) if x['kind']=='scalar' else a*a@st['membership'].T;k=32 if x['kind']=='scalar' else 8;gate=np.zeros_like(scores);np.put_along_axis(gate,np.argsort(scores,axis=-1)[...,-k:],1,axis=-1);mask=gate if x['kind']=='scalar' else gate@st['membership'];actual=initial['support'].numpy();mismatch=np.any(mask!=actual,axis=-1);ordered=np.sort(scores,axis=-1);gap=(ordered[...,-k]-ordered[...,-k-1])/np.maximum(abs(ordered[...,-k]),1e-12);assert np.all(gap[mismatch]<1e-5);ties+=int(mismatch.sum());mask[mismatch]=actual[mismatch];D=st['dictionary']/np.linalg.norm(st['dictionary'],axis=0,keepdims=True);g=spec['group'];atoms=np.arange(4*g,4*g+4);region=np.zeros_like(c['write'].numpy()[...,:1]);boundary=c['prefix_length'];region[:,:boundary]=1 if x['region']=='prefix' else 0;region[:,boundary:]=1 if x['region']=='task' else 0;delta=((a*mask)[...,atoms]@D[:,atoms].T)*scale*region;saved=sample['delta'].numpy();error=float(np.linalg.norm(delta-saved)/max(np.linalg.norm(saved),1e-20));decode_errors.append(error);assert error<1e-5
 n,nc,q,qc=[sample[k].numpy() for k in ['native','native_changed','intact','changed']];a,b,c,d=map(lp,[n,nc,q,qc]);nr=b-a;fr=d-c;R=float((np.exp(a)*(fr-nr)**2).sum());energy=float((np.exp(a)*nr**2).sum());groups=sample['groups'];v=sample['variant'];margin=lambda lg:float(lse(lg.astype(float)[groups[v]])-lse(lg.astype(float)[groups[1-v]]));nm=margin(n)-margin(nc);qm=margin(q)-margin(qc)
 for k,val in [('R',R),('native_response_energy',energy),('native_margin_effect',nm),('reconstructed_margin_effect',qm),('margin_squared_error',(qm-nm)**2)]:close(x[k],val);errors.append(abs(x[k]-val))
keys=set();directions={};oldjoint=json.loads((P/'finite_gradient_steps.json').read_text());oldcontrol=json.loads((P/'aggregate_kl_training_fit.json').read_text());Klookup={(x['target_index'],x['condition']):x['K'] for x in oldjoint['records'] if x['condition'].startswith('aggregate_gradient:')};Klookup.update({(x['target_index'],x['arm']):x['K'] for x in oldcontrol['records']})
for x in r['records']:
 i=x['target_index'];s=p['rows'][i];key=i,x['arm'];assert key not in keys;keys.add(key);assert all(x[k]==s[k] for k in ['kind','seed','family','region','group','input_key','token_hash']);inv=tuple(x[k] for k in ['noop','delta_energy','native_response_energy','native_margin_effect','native_margin_energy']);assert i not in directions or directions[i]==inv;directions[i]=inv
 if x['noop']:assert all(x[k]==0 for k in ['delta_energy','R','native_response_energy','margin_squared_error','native_margin_effect','reconstructed_margin_effect'])
 close(x['K'],t['records'][i]['initial_K'] if x['arm']=='initial' else Klookup[i,x['arm']]);close(x['native_margin_energy'],x['native_margin_effect']**2)
assert keys=={(i,e['arm']) for i,s in enumerate(p['rows']) for e in p['models'] if (e['kind'],e['seed'])==(s['kind'],s['seed'])};assert len(keys)==2400
unique_inputs=len({(x['kind'],x['seed'],x['input_key']) for x in t['records']});assert r['forward_count']==5*unique_inputs+6*sum(not x[0] for x in directions.values())
for s in r['summaries']:
 rows=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','arm']) and (s['family'] is None or x['family']==s['family']) and (s['region'] is None or x['region']==s['region'])];assert len(rows)==s['endpoints'];assert sum(not x['noop'] for x in rows)==s['nonzero'];close(sum(x['R'] for x in rows)/len(rows),s['R'])
 for key,sourcekey in [('native_response_energy_sum','native_response_energy'),('margin_error_sum','margin_squared_error'),('native_margin_energy_sum','native_margin_energy')]:close(sum(x[sourcekey] for x in rows),s[key])
 for val,num,den in [('relative_response_error',sum(x['R'] for x in rows),s['native_response_energy_sum']),('relative_margin_error',s['margin_error_sum'],s['native_margin_energy_sum'])]:
  if den:close(s[val],num/den)
  else:assert s[val] is None
out=dict(status='passed',result_sha256=sha(rp),selection_rows=480,models=20,records=len(keys),samples=len(samples),summaries=len(r['summaries']),noops=sum(x[0] for x in directions.values()),max_metric_error=max(errors),max_delta_relative_error=max(decode_errors),near_tie_rows=ties,scope='Independent exact-input/group-exclusion selection, NumPy sampled M18 decoder/delta/response metrics, all model hashes, old intact K replay, inventory/noops/forward-count and summaries; not independent native transformer implementation.');(P/'unseen_direction_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
