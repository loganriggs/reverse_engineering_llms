"""M69 all six encoded-change fit receipts, schedules, objective arithmetic and states."""
from pathlib import Path
import json,hashlib,collections
import torch,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);rp=V/'encoded_change_training_results.json';r=json.loads(rp.read_text());p=json.loads((V/'balanced_response_training_protocol.json').read_text());cal=json.loads((V/'encoded_change_calibration.json').read_text());receipt=r['receipt'];assert receipt['protocol_sha256']==sha(V/'encoded_change_training_protocol.json');assert receipt['evaluation_sha256']==sha(V/'encoded_change_evaluation_protocol.json');assert receipt['source_sha256']==sha(P/'train_encoded_change.py');assert receipt['loss_source_sha256']==sha(P/'encoded_change_loss.py');assert receipt['runtime_sha256']==sha(P/'balanced_training_runtime.py');assert receipt['cache_sha256']==json.loads((V/'encoded_change_calibration.json').read_text())['cache_sha256'];assert len(r['models'])==6
 for name,h in receipt['dependencies'].items():assert sha(V/name)==h
 expected={(x['kind'],x['seed'],a) for x in p['models'] for a in ['KL_encoded_balanced_response']};assert {(x['kind'],x['seed'],x['arm']) for x in r['models']}==expected;checks=[]
 for entry in r['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);initial=next(x for x in p['models'] if x['kind']==entry['kind'] and x['seed']==entry['seed']);ip=Path(initial['path']);assert sha(ip)==initial['sha256'];old=torch.load(ip,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'];assert len(s['history'])==132;assert torch.equal(s['mean'],old['mean']) and torch.equal(s['scale'],old['scale']);assert torch.equal(s['state']['membership'],old['state']['membership']);assert all(torch.isfinite(v).all() for v in s['state'].values());norm=s['state']['dictionary'].double().norm(dim=0);assert torch.allclose(norm,torch.ones_like(norm),rtol=1e-5,atol=1e-6);schedule=p['schedules'][str(entry['seed'])];counts=collections.Counter()
  for step,h in enumerate(s['history']):
   assert h['step']==step;parts=h['parts'];assert len(parts)==(1 if step%2==0 else 2)
   if step%2==0:
    assert parts[0]['key']==f"natural:{schedule['natural_documents'][step//2]}";assert parts[0]['loss']==parts[0]['K']
   else:
    task=schedule['task_updates'][step//2];counts[task['family']]+=1
    for ep,b in enumerate(parts):
     assert b['key']==task['pair_id']+f'|{ep}' and b['family']==task['family'];weight=cal['encoded_weight'];assert b['donor_key']==task['pair_id']+f'|{1-ep}' and 0<=b['donor_active_mean']<=32;assert b['weight']==weight;R=b['balanced_R'];assert np.isclose(b['loss'],(b['K']+weight*R)/2,rtol=1e-12,atol=1e-12)
   assert h['loss']==sum(b['loss'] for b in parts);assert all(np.isfinite(b['loss']) and 0<=b['active_mean']<=32 for b in parts)
  assert len(counts)==11 and set(counts.values())=={6};checks.append(dict(kind=entry['kind'],seed=entry['seed'],arm=entry['arm'],updates=132,natural_updates=66,task_pair_updates=66,max_column_norm_error=float((norm-1).abs().max()),first_online_loss=s['history'][0]['loss'],last_online_loss=s['history'][-1]['loss']))
 controls=json.loads((V/'balanced_response_training_results.json').read_text())['models']
 for ck in r['optimizer_default_checks']:
  ce=next(x for x in controls if (x['kind'],x['seed'],x['arm'])==(ck['kind'],ck['seed'],'intact_KL'));pp=Path(ce['path']).with_suffix('.progress.pt');assert sha(pp)==ck['control_progress_sha256'];oldopt=torch.load(pp,weights_only=True)['optimizer']['param_groups'][0];dummy=torch.nn.Parameter(torch.zeros(1));newopt=torch.optim.Adam([dummy],lr=p['learning_rate']).state_dict()['param_groups'][0];assert {k:v for k,v in oldopt.items() if k!='params'}=={k:v for k,v in newopt.items() if k!='params'}
 assert r['optimizer_updates']==792;assert len(r['optimizer_default_checks'])==6 and all(x['defaults_equal'] for x in r['optimizer_default_checks']);out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),fits=6,updates=792,checks=checks,scope='Every saved hash/receipt, all792update schedules/objective arithmetic, family counts, support limits and final state invariants. Online losses compare different examples and are not final retention/transfer. No independent optimizer trajectory replay.')
 op=V/'encoded_change_training_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',6,'fits',792,'updates')
if __name__=='__main__':main()
