"""Audit all final states and scheduled update records; not optimizer replay."""
from pathlib import Path
import json,hashlib,math
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
rp=P/'joint_shared_training_results.json';r=json.loads(rp.read_text());pp=P/'joint_shared_comparison_protocol.json';p=json.loads(pp.read_text());cp=P/'joint_shared_calibration.json';cal=json.loads(cp.read_text());receipt=r['receipt'];assert receipt['protocol_sha256']==sha(pp) and receipt['calibration_sha256']==sha(cp);assert receipt['audit_sha256']==sha(P/'joint_shared_calibration_audit.json');assert receipt['source_sha256']==sha(P.parent/'train_joint_shared_comparison.py');assert receipt['loss_source_sha256']==sha(P.parent/'joint_shared_loss.py')
assert sha(Path(r['natural_cache_path']))==r['natural_cache_sha256']
assign={(x['seed'],x['task_update'],x['endpoint']):x for x in p['target_assignments']};seen=set();checks=[]
for entry in r['models']:
 key=entry['kind'],entry['seed'],entry['arm'];assert key not in seen;seen.add(key);path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);assert s['receipt']==receipt;assert all(s[k]==entry[k] for k in ['kind','seed','arm']);initial=next(x for x in p['initial_models'] if (x['kind'],x['seed'])==key[:2]);assert s['initial_sha256']==initial['sha256']==sha(Path(initial['path']));old=torch.load(initial['path'],weights_only=True)
 assert torch.equal(s['mean'],old['mean']) and torch.equal(s['scale'],old['scale']);assert torch.equal(s['state']['membership'],old['state']['membership']);assert all(torch.isfinite(x).all() for x in s['state'].values());normerror=float((s['state']['dictionary'].norm(dim=0)-1).abs().max());assert normerror<1e-6
 assert len(s['history'])==120;sch=p['schedules'][str(key[1])]
 for step,h in enumerate(s['history']):
  assert h['step']==step;assert len(h['parts'])==(1 if step%2==0 else 2)
  for endpoint,part in enumerate(h['parts']):
   if step%2==0:assert part['key']==f"natural:{sch['natural_documents'][step]}";expected=[]
   else:
    a=assign[key[1],step//2,endpoint];assert part['key']==a['input_key'];expected=a['target_indices'] if key[2]=='intact_KL_plus_native_response' else []
   assert part['target_indices']==expected
   assert math.isclose(part['scaled_loss'],(part['K']+cal['response_weight']*part['R'])/len(h['parts']),rel_tol=1e-9,abs_tol=1e-10)
   if not expected:assert part['R']==0
   assert 0<=part['active_min']<=part['active_mean']<=part['active_max']<=32
   if key[0]!='shared_fixed':assert part['active_min']==part['active_max']==32
  assert math.isclose(h['loss'],sum(x['scaled_loss'] for x in h['parts']),rel_tol=1e-10,abs_tol=1e-10)
 changes={k:float((v.double()-old['state'][k].double()).norm()) for k,v in s['state'].items() if k!='membership'};assert sum(changes.values())>0
 checks.append(dict(kind=key[0],seed=key[1],arm=key[2],updates=len(s['history']),decoder_norm_max_error=normerror,parameter_changes=changes))
assert seen=={(kind,seed,arm) for kind in p['kinds'] for seed in p['seeds'] for arm in p['arms']};assert r['optimizer_updates']==1440
out=dict(status='passed',results_sha256=sha(rp),protocol_sha256=sha(pp),checks=checks,total_updates_checked=1440,scope='All final hashes/normalization/membership preservation, schedule keys, paired target sets and objective arithmetic. Does not independently replay optimization or establish final retention/generalization; those require evaluation.')
(P/'joint_shared_training_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(checks),'fits',1440,'updates')
