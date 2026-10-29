"""M61 all fixed fit receipts, update schedules, objective arithmetic and states."""
from pathlib import Path
import json,hashlib,collections
import torch,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);rp=V/'balanced_response_training_results.json';r=json.loads(rp.read_text());p=json.loads((V/'balanced_response_training_protocol.json').read_text());cal=json.loads((V/'balanced_response_calibration.json').read_text());receipt=r['receipt'];assert receipt['protocol_sha256']==sha(V/'balanced_response_training_protocol.json');assert receipt['calibration_sha256']==sha(V/'balanced_response_calibration.json');assert receipt['source_sha256']==sha(P/'train_balanced_response.py');assert receipt['task_loss_sha256']==sha(P/'balanced_response_loss.py');assert receipt['runtime_sha256']==sha(P/'balanced_training_runtime.py');assert receipt['natural_preflight_sha256']==sha(V/'balanced_natural_preflight.json');assert len(r['models'])==18
 expected={(x['kind'],x['seed'],a) for x in p['models'] for a in p['arms']};assert {(x['kind'],x['seed'],x['arm']) for x in r['models']}==expected;checks=[]
 for entry in r['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);initial=next(x for x in p['models'] if x['kind']==entry['kind'] and x['seed']==entry['seed']);ip=Path(initial['path']);assert sha(ip)==initial['sha256'];old=torch.load(ip,weights_only=True);assert s['receipt']==receipt and s['initial_sha256']==initial['sha256'];assert len(s['history'])==132;assert torch.equal(s['mean'],old['mean']) and torch.equal(s['scale'],old['scale']);assert torch.equal(s['state']['membership'],old['state']['membership']);assert all(torch.isfinite(v).all() for v in s['state'].values());norm=s['state']['dictionary'].double().norm(dim=0);assert torch.allclose(norm,torch.ones_like(norm),rtol=1e-5,atol=1e-6);schedule=p['schedules'][str(entry['seed'])];counts=collections.Counter()
  for step,h in enumerate(s['history']):
   assert h['step']==step;parts=h['parts'];assert len(parts)==(1 if step%2==0 else 2)
   if step%2==0:
    assert parts[0]['key']==f"natural:{schedule['natural_documents'][step//2]}";assert parts[0]['loss']==parts[0]['K']
   else:
    task=schedule['task_updates'][step//2];counts[task['family']]+=1
    for ep,b in enumerate(parts):
     assert b['key']==task['pair_id']+f'|{ep}' and b['family']==task['family'];weight=0 if entry['arm']=='intact_KL' else cal['raw_weight'] if entry['arm']=='KL_raw_response' else cal['balanced_weight'];assert b['weight']==weight;R=b['raw_R'] if entry['arm']=='KL_raw_response' else b['balanced_R'];assert np.isclose(b['loss'],(b['K']+weight*R)/2,rtol=1e-12,atol=1e-12)
   assert h['loss']==sum(b['loss'] for b in parts);assert all(np.isfinite(b['loss']) and 0<=b['active_mean']<=32 for b in parts)
  assert len(counts)==11 and set(counts.values())=={6};checks.append(dict(kind=entry['kind'],seed=entry['seed'],arm=entry['arm'],updates=132,natural_updates=66,task_pair_updates=66,max_column_norm_error=float((norm-1).abs().max()),first_online_loss=s['history'][0]['loss'],last_online_loss=s['history'][-1]['loss']))
 assert r['optimizer_updates']==2376;out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),fits=18,updates=2376,checks=checks,scope='Every saved hash/receipt, all2376update schedules/objective arithmetic, family counts, support limits and final state invariants. Online losses compare different examples and are not final retention/transfer. No independent optimizer trajectory replay.')
 op=V/'balanced_response_training_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',18,'fits',2376,'updates')
if __name__=='__main__':main()
