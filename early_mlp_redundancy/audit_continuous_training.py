"""M112 schedules, loss arithmetic, optimizer states, hard replay and final checkpoints."""
import json
from pathlib import Path
import torch
from fit_pca_encoder import V,sha


def main():
 torch.set_num_threads(1)
 rp=V/'continuous_training_results.json';r=json.loads(rp.read_text());p=json.loads((V/'continuous_training_protocol.json').read_text());base=json.loads((V/'balanced_response_training_protocol.json').read_text())
 assert r['receipt']['protocol_sha256']==sha(V/'continuous_training_protocol.json') and r['receipt']['runner_sha256']==sha(V.parent/'train_continuous_comparison.py')
 assert r['receipt']['evaluation_protocol_sha256']==sha(V/'continuous_training_evaluation_protocol.json')
 assert len(r['models'])==4 and {(x['seed'],x['mode']) for x in r['models']}=={(s,m) for s in [71,72] for m in ['hard','continuous']}
 previous=json.loads((V/'encoded_change_training_results.json').read_text())['models'];checks=[];maxarithmetic=0.;updates=0
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True);assert state['receipt']==r['receipt'];assert len(state['history'])==132
  initial=next(x for x in p['starts'] if x['seed']==entry['seed']);assert state['initial_sha256']==initial['sha256'];assert sha(initial['path'])==initial['sha256']
  schedule=base['schedules'][str(entry['seed'])];families=set();energy=[];active=[]
  for i,row in enumerate(state['history']):
   assert row['step']==i;parts=row['parts'];expectedloss=0.
   if i%2==0:
    assert len(parts)==1 and parts[0]['key']==f"natural:{schedule['natural_documents'][i//2]}";assert parts[0]['loss']==parts[0]['K'];expectedloss=parts[0]['K']
   else:
    assignment=schedule['task_updates'][i//2];assert len(parts)==2
    for ep,part in enumerate(parts):
     assert part['key']==assignment['pair_id']+f'|{ep}' and part['donor_key']==assignment['pair_id']+f'|{1-ep}' and part['family']==assignment['family'];families.add(part['family'])
     value=(part['K']+p['causal_weight']*part['balanced_R'])/2;err=abs(value-part['loss']);maxarithmetic=max(maxarithmetic,err);assert err<1e-8;expectedloss+=value
   assert abs(expectedloss-row['loss'])<1e-8
   for part in parts:assert 0<=part['active_mean']<=32 and part['code_energy']>=0;energy.append(part['code_energy']);active.append(part['active_mean'])
   updates+=1
  assert len(families)==11
  assert all(torch.isfinite(x).all() for x in state['state'].values());D=state['state']['dictionary'];assert torch.allclose(D.norm(dim=0),torch.ones(128),rtol=1e-5,atol=1e-6)
  progress=torch.load(Path(entry['path']).with_suffix('.progress.pt'),weights_only=True);assert progress['receipt']==r['receipt'] and len(progress['history'])==132
  assert all(torch.equal(t.cpu(),state['state'][name]) for name,t in progress['state'].items())
  assert all(int(x['step'])==132 for x in progress['optimizer']['state'].values())
  oldentry=next(x for x in previous if x['kind']=='block' and x['seed']==entry['seed']);assert sha(oldentry['path'])==oldentry['sha256'];old=torch.load(oldentry['path'],weights_only=True)
  oldprogress=torch.load(Path(oldentry['path']).with_suffix('.progress.pt'),weights_only=True)
  normalize=lambda x:{k:v for k,v in x['param_groups'][0].items() if k!='params'}
  assert normalize(progress['optimizer'])==normalize(oldprogress['optimizer'])
  exact=all(torch.equal(state['state'][k],old['state'][k]) for k in old['state']) if entry['mode']=='hard' else None
  if entry['mode']=='hard':assert exact==entry['hard_replay']['bit_exact']
  checks.append(dict(seed=entry['seed'],mode=entry['mode'],updates=132,training_families=sorted(families),optimizer_defaults_match_historical=True,hard_reference_bit_exact=exact,active_range=[min(active),max(active)],online_code_energy_range=[min(energy),max(energy)],first_online_loss=state['history'][0]['loss'],last_online_loss=state['history'][-1]['loss']))
 assert updates==528
 out=dict(status='passed',source_sha256=sha(__file__),results_sha256=sha(rp),updates=updates,max_loss_arithmetic_error=maxarithmetic,checks=checks,scope='Accounting/final state audit, not independent replay of continuous optimization trajectory or held-out evaluation. First/last online losses use different examples and are not a convergence measure.')
 dest=V/'continuous_training_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
