"""M98 training-only causal coverage inventory; no fitting or model forward."""
import hashlib,json
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import V,sha

def digest(ids):return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()

def main():
 torch.set_num_threads(1);design=json.loads((V/'product_dictionary_coverage_design.json').read_text())
 for name,value in design['references'].items():assert sha(V/name)==value
 inventory=json.loads((V/'input_training_inventory.json').read_text());assert sha(inventory['path'])==inventory['sha256'];inputs=torch.load(inventory['path'],weights_only=True)['inputs'];protocol=json.loads((V/'training_causal_coverage_protocol.json').read_text());pairs={p['id']:p for p in protocol['pairs']};records=json.loads((V/'training_causal_coverage.json').read_text())['records'];targets={(r['pair_id'],r['endpoint'],r['region']):r for r in records if r['layer']==1};competence={p['pair_id']:p for p in inventory['pairs']};old=json.loads((V/'balanced_response_training_protocol.json').read_text())['schedules'];assert len(pairs)==70 and len(inputs)==140
 rows=[];prefix_vectors=defaultdict(list)
 for pid,pair in pairs.items():
  endpoints=[]
  for ep in [0,1]:
   key=f'{pid}|{ep}';ids=inputs[key]['ids'].reshape(-1).tolist();assert ids==pair['token_ids'][ep];z=inputs[key]['input'].reshape(-1,1152);assert len(z)==len(ids);sites={}
   for region in ['changed_input_positions','final_position','identity']:
    target=targets[pid,ep,region]
    if target['status']!='evaluated':sites[region]=dict(status=target['status'],reason=target['reason'],prefixes=[],input_hashes=[]);continue
    prefixes=[];input_hashes=[]
    for position in target['recipient_positions']:
     prefix=digest(ids[:position+1]);vector=z[position].contiguous().numpy();vh=hashlib.sha256(vector.tobytes()).hexdigest();prefixes.append(prefix);input_hashes.append(vh);prefix_vectors[prefix].append(vector)
    sites[region]=dict(status='evaluated',prefixes=prefixes,input_hashes=input_hashes,positions=target['recipient_positions'])
   endpoints.append(dict(endpoint=ep,full_prompt_hash=digest(ids),sites=sites))
  rows.append(dict(pair_id=pid,family=pair['family'],both_contrast=competence[pid]['both_contrast'],both_top1=competence[pid]['both_top1'],endpoints=endpoints))
 def summarize(selected,family=None):
  rr=[r for r in rows if r['pair_id'] in selected and (family is None or r['family']==family)];endpoints=[e for r in rr for e in r['endpoints']];result=dict(pairs=len(rr),endpoints=len(endpoints),distinct_prompts=len({e['full_prompt_hash'] for e in endpoints}),both_contrast_pairs=sum(r['both_contrast'] for r in rr),both_top1_pairs=sum(r['both_top1'] for r in rr),sites={})
  for region in ['changed_input_positions','final_position','identity']:
   ss=[e['sites'][region] for e in endpoints];result['sites'][region]=dict(evaluated_endpoints=sum(s['status']=='evaluated' for s in ss),ineligible_endpoints=sum(s['status']!='evaluated' for s in ss),positions=sum(len(s['prefixes']) for s in ss),distinct_prefixes=len({h for s in ss for h in s['prefixes']}),distinct_bitwise_inputs=len({h for s in ss for h in s['input_hashes']}))
  return result
 summaries=[]
 for seed in [71,72]:
  visited={x['pair_id'] for x in old[str(seed)]['task_updates']};assert visited<=set(pairs)
  for family in [None]+sorted({p['family'] for p in pairs.values()}):
   full=summarize(set(pairs),family);subset=summarize(visited,family);summaries.append(dict(seed=seed,family=family,full_pool=full,old_visited=subset,additional_source_prefixes=full['sites']['changed_input_positions']['distinct_prefixes']-subset['sites']['changed_input_positions']['distinct_prefixes'],additional_final_prefixes=full['sites']['final_position']['distinct_prefixes']-subset['sites']['final_position']['distinct_prefixes']))
 # Same token prefix is the causal-context key; tiny hardware-dependent bit
 # differences must not be sold as new causal examples.
 repeated=0;different_bits=0;maximum=0.
 for vectors in prefix_vectors.values():
  if len(vectors)<2:continue
  repeated+=1;base=vectors[0].astype(float)
  for vector in vectors[1:]:
   different_bits+=not np.array_equal(vector,vectors[0]);relative=float(np.linalg.norm(vector.astype(float)-base)/max(np.linalg.norm(base),1e-30));maximum=max(maximum,relative)
 excluded=set(json.loads((V/'diverse_supervision_split.json').read_text())['evaluation_pair_ids']);assert not set(pairs)&excluded
 whole=json.loads((V/'whole_task_native_protocol.json').read_text())['pairs'];source=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];assert not {p['family'] for p in pairs.values()}&{p['family'] for p in whole};prompts={e['full_prompt_hash'] for r in rows for e in r['endpoints']};assert not prompts&{digest(ids) for p in whole+source for ids in p['token_ids']}
 out=dict(status='coverage_measured',source_sha256=sha(__file__),design_sha256=sha(V/'product_dictionary_coverage_design.json'),input_inventory_sha256=sha(V/'input_training_inventory.json'),rows=rows,summaries=summaries,repeated_prefix_groups=repeated,bitwise_differences_within_identical_prefix=different_bits,max_relative_input_difference_within_prefix=maximum,source_holdout_pairs_excluded=len(excluded),whole_task_families_excluded=len({p['family'] for p in whole}),scope='Traininginput/metadata only. Prefix counts identify distinct causal contexts, not independent semantic manifolds. Bitwise input variation at identical prefixes is reported separately. No heldout outcomes or gradients used to select data, no newnativeforwards.')
 path=V/'product_dictionary_coverage_inventory.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
 for s in summaries:
  if s['family'] is None:print(s,flush=True)
 print('PREFIX_NUMERICS',repeated,different_bits,maximum,flush=True)
if __name__=='__main__':main()
