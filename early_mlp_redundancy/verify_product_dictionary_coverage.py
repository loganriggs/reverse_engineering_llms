"""Independent tuple/array coverage and matched schedule verification."""
import json
from collections import Counter
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);ip=V/'product_dictionary_coverage_inventory.json';inv=json.loads(ip.read_text());assert inv['source_sha256']==sha(V.parent/'audit_product_dictionary_coverage.py');rp=V/'product_dictionary_coverage_schedules.json';r=json.loads(rp.read_text());assert r['source_sha256']==sha(V.parent/'freeze_product_dictionary_coverage_schedules.py') and r['inventory_sha256']==sha(ip)
 source=json.loads((V/'input_training_inventory.json').read_text());assert sha(source['path'])==source['sha256'];inputs=torch.load(source['path'],weights_only=True)['inputs'];pairs={p['id']:p for p in json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs']};records={(x['pair_id'],x['endpoint'],x['region']):x for x in json.loads((V/'training_causal_coverage.json').read_text())['records'] if x['layer']==1};old=json.loads((V/'balanced_response_training_protocol.json').read_text())['schedules'];rows=[]
 def counts(selected,family,region):
  prefixes=set();vectors=set();eligible=invalid=positions=0
  for pid in selected:
   if family is not None and pairs[pid]['family']!=family:continue
   for ep in [0,1]:
    record=records[pid,ep,region];z=inputs[f'{pid}|{ep}']['input'].reshape(-1,1152);ids=tuple(pairs[pid]['token_ids'][ep]);assert tuple(inputs[f'{pid}|{ep}']['ids'].reshape(-1).tolist())==ids
    if record['status']!='evaluated':invalid+=1;continue
    eligible+=1
    for pos in record['recipient_positions']:positions+=1;prefixes.add(ids[:pos+1]);vectors.add(tuple(z[pos].tolist()))
  return dict(evaluated_endpoints=eligible,ineligible_endpoints=invalid,positions=positions,distinct_prefixes=len(prefixes),distinct_bitwise_inputs=len(vectors))
 for summary in inv['summaries']:
  seed=summary['seed'];family=summary['family'];visited={x['pair_id'] for x in old[str(seed)]['task_updates']}
  for name,selected in [('full_pool',set(pairs)),('old_visited',visited)]:
   for region in ['changed_input_positions','final_position','identity']:assert counts(selected,family,region)==summary[name]['sites'][region]
 for seed in [71,72]:
  full=r['schedules'][str(seed)]['full_pool'];control=r['schedules'][str(seed)]['old_visited'];visited={x['pair_id'] for x in old[str(seed)]['task_updates']};assert len(full['task_updates'])==len(control['task_updates'])==len(full['natural_documents'])==70;assert len({x['pair_id'] for x in full['task_updates']})==70;assert {x['pair_id'] for x in full['task_updates']}==set(pairs);assert {x['pair_id'] for x in control['task_updates']}==visited;assert full['natural_documents']==control['natural_documents']==old[str(seed)]['natural_documents']+old[str(seed)]['natural_documents'][:4];assert all(0<=doc<32 for doc in full['natural_documents']);assert [x['family'] for x in full['task_updates']]==[x['family'] for x in control['task_updates']]
  coverage=next(x for x in inv['summaries'] if x['seed']==seed and x['family'] is None);assert coverage['additional_source_prefixes']>0 and coverage['additional_final_prefixes']>0
  for condition,schedule in [('full_pool',full),('old_visited',control)]:
   for a in schedule['task_updates']:assert pairs[a['pair_id']]['family']==a['family']
   assigned=[x['pair_id'] for x in schedule['task_updates']];prefixes=set();unavailable=0;tokens=0
   for pid in assigned:
    for ep in [0,1]:
     ids=pairs[pid]['token_ids'][ep];tokens+=len(ids);target=records[pid,ep,'changed_input_positions']
     if target['status']!='evaluated':unavailable+=1
     else:
      for pos in target['recipient_positions']:prefixes.add(tuple(ids[:pos+1]))
   rows.append(dict(seed=seed,condition=condition,scheduled_pairs=70,unique_pairs=len(set(assigned)),family_counts=dict(Counter(pairs[pid]['family'] for pid in assigned)),source_prefixes=len(prefixes),source_ineligible_endpoint_assignments=unavailable,scheduled_endpoint_tokens=tokens))
 excluded=set(json.loads((V/'diverse_supervision_split.json').read_text())['evaluation_pair_ids']);whole=json.loads((V/'whole_task_native_protocol.json').read_text())['pairs'];sourcehold=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];assert not set(pairs)&excluded;assert not {p['family'] for p in pairs.values()}&{p['family'] for p in whole};trainprompts={tuple(ids) for p in pairs.values() for ids in p['token_ids']};assert not trainprompts&{tuple(ids) for p in whole+sourcehold for ids in p['token_ids']}
 out=dict(status='passed',source_sha256=sha(__file__),inventory_sha256=sha(ip),schedules_sha256=sha(rp),summaries_verified=len(inv['summaries']),rows=rows,scope='Independent rawprefix tuples and normalized-input tuples reproduce everyfamily/site count. Exactfamily/natural sequences andcasebudgets matched. Split exclusions verified. Tokenwork andsourceavailability notidentical; preserved as designlimitations, notmatched-compute claims.')
 path=V/'product_dictionary_coverage_audit.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
