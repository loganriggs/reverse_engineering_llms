"""M16 sampled region decoding/output audit and complete intervention accounting."""
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
ip=P/'context_region_roles.json';r=json.loads(ip.read_text());pp=P/'context_region_protocol.json';protocol=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];deltaerrors=[]
for s in samples:
 row=r['records'][s['row']];ref=s['intact'].numpy();out=s['ablated'].numpy();lp=logsoft(ref);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(ref.argmax(-1)==out.argmax(-1)).mean());atoms=s['atoms'].numpy();delta=(s['code'].numpy()[...,atoms].astype(float)@s['D'].numpy()[:,atoms].astype(float).T)*float(s['scale'])*s['mask'].numpy();saved=s['contribution'].numpy();norm=max(np.linalg.norm(delta),1e-20);deltaerrors.append(float(np.linalg.norm(delta-saved)/norm));assert deltaerrors[-1]<1e-6;assert np.array_equal((s['q']-s['contribution']).numpy(),s['changed'].numpy());energy=np.sum(saved.astype(float)**2)/np.sum(s['q'].numpy().astype(float)**2);vals['removed_relative_energy']=energy
 for k,v in vals.items():errors.append(abs(float(v)-row[k]));assert errors[-1]<1e-7
 margins=[]
 for field,lg in [('intact_task',ref),('ablated_task',out)]:
  x=lg[0,-1].astype(float);g=s['groups'];v=s['variant'];margin=lse(x[g[v]])-lse(x[g[1-v]]);assert abs(margin-row[field]['margin'])<1e-10;margins.append(margin)
 assert abs(margins[0]-margins[1]-row['margin_loss'])<1e-10
assert len(r['records'])==20736
keys=set()
for row in r['records']:
 key=tuple(row[k] for k in ['id','endpoint','kind','seed','grouping','group','region']);assert key not in keys;keys.add(key)
 if row['noop']:assert row['margin_loss']==0 and row['mean_KL']==0 and row['mean_removed_active_coordinates']==0
screen=json.loads((P/'context_competence.json').read_text());old=json.loads((P/'group_role_matching.json').read_text());assert sha(P/'group_role_matching.json')==protocol['matching_sha256']
for b in r['baselines']:
 native=next(x for x in screen['results'] if x['context']==b['context'] and x['id']==b['base_id'])['endpoints'][b['endpoint']];assert abs(native['margin']-b['native_task']['margin'])<1e-10
whole=json.loads((P/'context_group_roles.json').read_text());assert sha(P/'context_group_roles.json')==protocol['all_position_reference_sha256']
base={(x['id'],x['endpoint'],x['kind'],x['seed']):x for x in whole['baselines']}
for x in r['baselines']:
 b=base[(x['id'],x['endpoint'],x['kind'],x['seed'])]
 assert abs(x['task']['margin']-b['task']['margin'])<1e-8
for s in samples:
 row=r['records'][s['row']];mask=s['mask'].numpy();n=row['prefix_length'];assert n==protocol['prefix_lengths'][row['id']]
 expected=np.zeros_like(mask);expected[:,:n]=1 if row['region']=='prefix' else 0;expected[:,n:]=1 if row['region']=='task' else 0
 assert np.array_equal(mask,expected)
 # The unmasked decoder contribution is exactly partitioned by these complementary masks.
 full=(s['code'].numpy()[...,s['atoms'].numpy()].astype(float)@s['D'].numpy()[:,s['atoms'].numpy()].astype(float).T)*float(s['scale'])
 assert np.allclose(full*mask+full*(1-mask),full)
counts={region:sum(x['region']==region for x in r['records']) for region in ['prefix','task']};assert counts=={'prefix':10368,'task':10368}
out=dict(status='passed',ablation_samples=len(samples),unique_records=len(keys),region_counts=counts,native_replays=len(r['baselines']),max_metric_error=max(errors),max_decode_relative_error=max(deltaerrors),records_sha256=sha(ip),scope='Sampled independent full-output metrics and masked decoding; every key/noop and native/intact replay; region masks/complementarity. Separate analysis audit required for role matrices and summaries.')
(P/'context_region_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
