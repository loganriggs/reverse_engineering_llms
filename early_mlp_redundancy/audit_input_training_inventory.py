"""M80 independent cached-input polynomial replay and inventory audit."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V,A

def main():
 torch.set_num_threads(1);rp=V/'input_training_inventory.json';r=json.loads(rp.read_text());assert sha(r['path'])==r['sha256'];s=torch.load(r['path'],weights_only=True);old=torch.load(A/'emlp_m60_training_cache.pt',weights_only=True);assert sha(A/'emlp_m60_training_cache.pt')==s['old_cache_sha256'];ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];weights=torch.load(ref['path'],weights_only=True);L,R,D=[weights[k].double().numpy() for k in ['L','R','D']];pairs=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'];expected={p['id']+f'|{ep}' for p in pairs for ep in [0,1]};assert set(s['inputs'])==expected==set(old['inputs']);maximum=0
 for key,c in s['inputs'].items():
  assert torch.equal(c['ids'],old['inputs'][key]['ids']) and c['family']==old['inputs'][key]['family'];z=c['input'].double().numpy();w=old['inputs'][key]['write'].double().numpy();prediction=((z@L.T)*(z@R.T))@D.T;error=np.linalg.norm(prediction-w)/max(1,np.linalg.norm(w));assert error<1e-5;maximum=max(maximum,float(error))
 records={(x['pair_id'],x['endpoint']):x for x in json.loads((V/'training_causal_coverage.json').read_text())['records'] if x['layer']==1 and x['region']=='final_position'}
 for row in r['pairs']:
  ep=[records[row['pair_id'],i] for i in [0,1]];assert row['both_contrast']==all(x['native_binary'] for x in ep) and row['both_top1']==all(x['native_top1'] for x in ep)
 split=json.loads((V/'diverse_supervision_split.json').read_text());assert {p['id'] for p in pairs}.isdisjoint(split['evaluation_pair_ids']);held=json.loads((V/'whole_task_native_protocol.json').read_text())['pairs'];assert {p['family'] for p in pairs}.isdisjoint(p['family'] for p in held);assert {tuple(ids) for p in pairs for ids in p['token_ids']}.isdisjoint(tuple(ids) for p in held for ids in p['token_ids'])
 out=dict(status='passed',source_sha256=sha(__file__),inventory_sha256=sha(rp),endpoints=140,max_polynomial_relative_error=maximum,both_contrast_pairs=sum(x['both_contrast'] for x in r['pairs']),both_top1_pairs=sum(x['both_top1'] for x in r['pairs']),source_group_holdouts=len(split['evaluation_pair_ids']),whole_task_exclusions=len({p['family'] for p in held}),scope='Independent NumPy polynomial replay of every captured input; exact ID/family inventory, competence and current holdout exclusions checked. Does not establish semantic absence from language-model pretraining or replay causal teachers again.')
 dest=V/'input_training_inventory_audit.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
