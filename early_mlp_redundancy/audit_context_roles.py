"""M13 NumPy ablation metrics and independent role-prediction accounting."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def lse(x):
 m=x.max();return float(m+np.log(np.exp(x-m).sum()))
ip=P/'context_group_roles.json';r=json.loads(ip.read_text());mp=P/'context_role_transfer.json';m=json.loads(mp.read_text());assert sha(ip)==m['group_roles_sha256'];assert sha(P/'context_group_roles_protocol.json')==r['protocol_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];deltaerrors=[]
for s in samples:
 row=r['records'][s['row']];ref=s['intact'].numpy();out=s['ablated'].numpy();lp=logsoft(ref);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(ref.argmax(-1)==out.argmax(-1)).mean());atoms=s['atoms'].numpy();delta=(s['code'].numpy()[...,atoms].astype(float)@s['D'].numpy()[:,atoms].astype(float).T)*float(s['scale']);saved=s['contribution'].numpy();norm=max(np.linalg.norm(delta),1e-20);deltaerrors.append(float(np.linalg.norm(delta-saved)/norm));assert deltaerrors[-1]<1e-6;assert np.array_equal((s['q']-s['contribution']).numpy(),s['changed'].numpy());energy=np.sum(saved.astype(float)**2)/np.sum(s['q'].numpy().astype(float)**2);vals['removed_relative_energy']=energy
 for k,v in vals.items():errors.append(abs(float(v)-row[k]));assert errors[-1]<1e-7
 margins=[]
 for field,lg in [('intact_task',ref),('ablated_task',out)]:
  x=lg[0,-1].astype(float);g=s['groups'];v=s['variant'];margin=lse(x[g[v]])-lse(x[g[1-v]]);assert abs(margin-row[field]['margin'])<1e-10;margins.append(margin)
 assert abs(margins[0]-margins[1]-row['margin_loss'])<1e-10
assert len(r['records'])==10368
keys=set()
for row in r['records']:
 key=tuple(row[k] for k in ['id','endpoint','kind','seed','grouping','group']);assert key not in keys;keys.add(key)
 if row['noop']:assert row['margin_loss']==0 and row['mean_KL']==0 and row['mean_removed_active_coordinates']==0
screen=json.loads((P/'context_competence.json').read_text());old=json.loads((P/'group_role_matching.json').read_text());assert sha(P/'group_role_matching.json')==m['frozen_matching_sha256']
for b in r['baselines']:
 native=next(x for x in screen['results'] if x['context']==b['context'] and x['id']==b['base_id'])['endpoints'][b['endpoint']];assert abs(native['margin']-b['native_task']['margin'])<1e-10
for result in m['results']:
 context=result['context'];kind=result['kind'];grouping=result['grouping'];families=result['families'];expected=sorted(x['family'] for x in screen['cohorts'] if x['context']==context and x['pass_gate']);assert families==expected;match=next(x for x in old['matches'] if x['kind']==kind and x['grouping']==grouping and x['method']==result['method']);assert result['assignment']==match['assignment'] and result['amplitudes']==match['train_amplitudes'];arrays=[]
 for seed in [71,72]:
  mat=np.zeros((32,len(families)));counts=np.zeros_like(mat)
  for row in r['records']:
   if row['context']==context and row['kind']==kind and row['grouping']==grouping and row['seed']==seed:
    g=row['group'];j=families.index(row['family']);mat[g,j]+=row['margin_loss'];counts[g,j]+=1
  assert (counts>0).all();mat/=counts;assert np.max(np.abs(mat-np.array(m['matrices'][f'{context}:{kind}:{grouping}:{seed}'])))<1e-12;arrays.append(mat)
 idx=[old['families']['evaluation'].index(f) for f in families];oldarrays=[np.array(old['signatures'][f'{kind}:{grouping}:{seed}:evaluation']['margin'])[:,idx] for seed in [71,72]]
 for label,(a,b) in [('current',arrays),('original_same_families',oldarrays)]:
  target=b[result['assignment']];pred=a*np.array(result['amplitudes'])[:,None];energy=(target*target).sum(0);err=((pred-target)**2).sum(0);assert abs(float(err.sum()/energy.sum())-result[label]['relative_squared_error'])<1e-10;assert abs(float((err/energy).mean())-result[label]['equal_family_mean_relative_error'])<1e-10
out=dict(status='passed',ablation_samples=len(samples),unique_intervention_records=len(keys),native_replay_baselines=len(r['baselines']),frozen_matching_scores=len(m['results']),max_metric_absolute_error=max(errors),max_group_decode_relative_error=max(deltaerrors),group_roles_sha256=sha(ip),transfer_sha256=sha(mp),scope='Independent sampledecoding/outputscores,allnoops/uniqueness,nativecompetencereplay,frozenassignments/amplitudes,familymatrices/currentandmatchedfamilyoldscoreaccounting.')
(P/'context_roles_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
