"""M98 exact inventory, schedule, optimizer and immutable-state audit."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V

def main():
 torch.set_num_threads(1);rp=V/'product_dictionary_coverage_training_results.json';r=json.loads(rp.read_text());pp=V/'product_dictionary_coverage_training_protocol.json';p=json.loads(pp.read_text());assert r['receipt']['protocol_sha256']==sha(pp) and r['receipt']['source_sha256']==sha(V.parent/'train_product_dictionary_coverage.py');schedules=json.loads((V/'product_dictionary_coverage_schedules.json').read_text())['schedules'];expected={(m['kind'],m['seed'],m['coverage'],arm) for m in p['models'] for arm in p['arms']};assert {(e['kind'],e['seed'],e['coverage'],e['arm']) for e in r['models']}==expected and len(r['models'])==8;rows=[]
 for name,digest in p['references'].items():assert sha(V/name)==digest
 assert p['runner_sha256']==sha(V.parent/'train_product_dictionary_coverage.py')
 for entry in r['models']:
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True);initial=next(m for m in p['models'] if (m['kind'],m['seed'],m['coverage'])==(entry['kind'],entry['seed'],entry['coverage']));assert s['receipt']==r['receipt'] and s['initial_sha256']==initial['sha256'] and sha(initial['path'])==initial['sha256'];start=torch.load(initial['path'],weights_only=True);assert s['optimizer_updates']==len(s['history'])==140;progress=torch.load(entry['path'].replace('.pt','.progress.pt'),weights_only=True);assert progress['receipt']==s['receipt'] and progress['history']==s['history'];assert all(torch.equal(v,progress['state'][k].cpu()) for k,v in s['state'].items());assert len(progress['optimizer']['state'])==1;assert all(int(v['step'])==140 for v in progress['optimizer']['state'].values());assert s['optimizer_defaults']['lr']==.0001 and tuple(s['optimizer_defaults']['betas'])==(.9,.999) and s['optimizer_defaults']['eps']==1e-8
  assert s['coverage']==entry['coverage'] and s['seed']==entry['seed'] and s['arm']==entry['arm']
  for key in ['left','right','rms','dictionary','offset']:assert torch.equal(s['state'][key],start['state'][key])
  assert all(torch.isfinite(v).all() for v in s['state'].values());assert torch.allclose(s['state']['dictionary'].norm(dim=0),torch.ones(128),rtol=1e-5,atol=1e-6);schedule=schedules[str(entry['seed'])][entry['coverage']];parts=0
  for step,record in enumerate(s['history']):
   assert record['step']==step;rr=record['parts'];assert len(rr)==(1 if step%2==0 else 2)
   for ep,row in enumerate(rr):
    if step%2==0:assert row['key']==f"natural:{schedule['natural_documents'][step//2]}" and row['B'] is None;want=row['K']
    else:
     assignment=schedule['task_updates'][step//2];assert row['key']==assignment['pair_id']+f'|{ep}' and row['family']==assignment['family'];want=row['K']/2
     if entry['arm']=='joint_three_corner':assert row['B'] is not None;want+=p['joint_weight']*row['B']/2
     else:assert row['B'] is None
    assert np.isfinite(row['loss']) and np.isclose(row['loss'],want,rtol=1e-12,atol=1e-12);parts+=1
   assert np.isclose(sum(x['loss'] for x in rr),record['loss'],rtol=1e-12,atol=1e-12)
  rows.append(dict(kind=entry['kind'],seed=entry['seed'],coverage=entry['coverage'],arm=entry['arm'],updates=140,loss_parts=parts,immutable_buffers_equal=True,parameters_changed=not torch.equal(s['state']['coefficients'],start['state']['coefficients'])))
 out=dict(status='passed_structural_and_schedule_audit',source_sha256=sha(__file__),results_sha256=sha(rp),models=rows,optimizer_updates=1120,scope='All8starts/finalstates/Adamstepcounts/defaults/buffers and everyloss-part arithmetic/schedule checked. Does not replay optimizertrajectory or independently score modeloutputs; fixed native final-loss audit remains separate.')
 dest=V/'product_dictionary_coverage_training_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('AUDITED',len(rows),1120)
if __name__=='__main__':main()
