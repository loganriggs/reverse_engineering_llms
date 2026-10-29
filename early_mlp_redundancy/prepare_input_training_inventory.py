"""M80 verify training exclusions and capture normalized MLP1 inputs only."""
import json
from collections import Counter,defaultdict
import torch
from inspect_model import load_model,trace
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'input_training_inventory_protocol.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 train=json.loads((V/'training_causal_coverage_protocol.json').read_text());native=json.loads((V/'training_causal_coverage.json').read_text());excluded=json.loads((V/'whole_task_native_protocol.json').read_text());split=json.loads((V/'diverse_supervision_split.json').read_text());pairs=train['pairs'];assert len(pairs)==70
 ids={p['id'] for p in pairs};assert set(split['train_pair_ids']).issubset(ids) and not ids.intersection(split['evaluation_pair_ids']);assert len(ids-set(split['train_pair_ids']))==4
 tf={p['family'] for p in pairs};ef={p['family'] for p in excluded['pairs']};assert tf.isdisjoint(ef)
 prompts={tuple(t) for p in pairs for t in p['token_ids']};evalprompts={tuple(t) for p in excluded['pairs'] for t in p['token_ids']};assert prompts.isdisjoint(evalprompts)
 cachepath=A/'emlp_m60_training_cache.pt';oldreceipt=json.loads((V/'encoded_change_training_receipt.json').read_text());assert sha(cachepath)==oldreceipt['cache_sha256'];old=torch.load(cachepath,weights_only=True);assert len(old['inputs'])==140
 records={(r['pair_id'],r['endpoint'],r['region']):r for r in native['records'] if r['layer']==1};families=defaultdict(list);model,receipt=load_model();mlp=model.transformer.h[1].mlp;captures={};calls=0
 for pair in pairs:
  endpoints=[]
  for ep in [0,1]:
   key=pair['id']+f'|{ep}';c=old['inputs'][key];assert c['ids'][0].tolist()==pair['token_ids'][ep];ids_tensor=c['ids'].cuda();out,cap=trace(model,ids_tensor);calls+=1;assert torch.equal(out[0,-1].cpu(),c['native']);w=mlp.Down(cap[1]['products']);assert torch.equal(w.cpu(),c['write']);z=cap[1]['normalized_input'].cpu();assert z.shape==w.shape and torch.isfinite(z).all();captures[key]=dict(ids=c['ids'],input=z,family=pair['family'])
   r=records[pair['id'],ep,'final_position'];endpoints.append(dict(binary=r['native_binary'],top1=r['native_top1']))
  families[pair['family']].append(dict(pair_id=pair['id'],both_contrast=all(e['binary'] for e in endpoints),both_top1=all(e['top1'] for e in endpoints),endpoints=endpoints,source_eligible=pair['source_region_eligible']))
 path=A/'emlp_m80_training_inputs.pt';assert not path.exists();torch.save(dict(protocol_sha256=sha(pp),source_sha256=sha(__file__),checkpoint_sha256=receipt.weights_sha256,old_cache_sha256=sha(cachepath),inputs=captures),path)
 summaries=[]
 for family,rows in sorted(families.items()):summaries.append(dict(family=family,pairs=len(rows),both_contrast=sum(r['both_contrast'] for r in rows),both_top1=sum(r['both_top1'] for r in rows),source_eligible=sum(r['source_eligible'] for r in rows),unique_prefixes_at_source=len({tuple(p['token_ids'][ep][:p['changed_positions'][0]+1]) for p in pairs if p['family']==family and p['source_region_eligible'] and p['changed_positions'] for ep in [0,1]})))
 result=dict(status='captured_verified',source_sha256=sha(__file__),protocol_sha256=sha(pp),path=str(path),sha256=sha(path),forward_count=calls,training_pairs=70,endpoints=140,source_group_holdout_pairs=len(split['evaluation_pair_ids']),whole_task_holdout_families=sorted(ef),family_overlap=[],exact_token_prompt_overlap=[],cached_targets={reg:sum(r['layer']==1 and r['region']==reg and r['status']=='evaluated' for r in native['records']) for reg in ['changed_input_positions','final_position','identity']},families=summaries,pairs=[r for rows in families.values() for r in rows],old_cache_has_normalized_input=False,scope='Only normalized inputs added. All140 native writes and final logits exactly replay historical cache. Existing cached causal targets retained, not newly replayed here. No featurizer fitting or evaluation inputs captured. Competence is measured contrast/top1, not free-form mastery; source-prefix counts expose repeated effective contexts.')
 op=V/'input_training_inventory.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
