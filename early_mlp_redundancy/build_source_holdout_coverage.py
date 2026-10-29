"""M59 freeze existing source-group development holdouts, no new selection."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 panel=json.loads((V/'training_source_context_audit.json').read_text());split=json.loads((V/'diverse_supervision_split.json').read_text());train=set(split['evaluation_pair_ids']);pairs=[]
 for p in panel['rows']:
  if p['id'] in train:pairs.append({k:p[k] for k in ['id','family','token_ids','answer_groups','variants']}|dict(source='existing_source_group_holdout'))
 assert len(pairs)==14
 exclusions={'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense','continent_recall','object_function'};assert not exclusions&{p['family'] for p in pairs};assert not set(split['train_pair_ids'])&{p['id'] for p in pairs};assignments=[]
 for p in pairs:
  a,b=p['token_ids'];equal=len(a)==len(b);changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y] if equal else [];p.update(equal_length=equal,changed_positions=changed,source_region_eligible=equal and bool(changed),source_ineligibility_reason=None if equal and changed else 'unequal_length' if not equal else 'identical_token_sequence')
  for ep in [0,1]:
   for layer in [1]:
    for region in ['changed_input_positions','final_position','identity']:
     eligible=region!='changed_input_positions' or p['source_region_eligible'];positions=changed if region=='changed_input_positions' else [len(p['token_ids'][ep])-1] if region=='final_position' else [];assignments.append(dict(pair_id=p['id'],endpoint=ep,layer=layer,region=region,eligible=eligible,recipient_positions=positions,donor_positions=changed if region=='changed_input_positions' else [len(p['token_ids'][1-ep])-1] if region=='final_position' else []))
 out=dict(stage='frozen_before_native_outcomes',source_sha256=sha(Path(__file__)),input_hashes={n:sha(V/n) for n in ['training_source_context_audit.json','diverse_supervision_split.json','coverage_source_holdout_design.json']},pairs=pairs,assignments=assignments,layers=[1],training_exclusions=sorted(exclusions),semantics='Replace native product write at selected positions with opposite endpoint write at aligned addresses, preserving Down bias. Changed-token positions use index equality only for equal-length prompts: this is an address-based contrast, not semantic alignment. Final positions align explicitly even with unequal lengths. Donors may change multiple linguistic variables. Intervene one layer/region at a time; no simultaneous layer patch.',metrics='Final full-vocabulary native-weighted squared logprob response, output KL, intended-vs-alternate margin change, task competence and raw delta energy. No norm matching or amplitudes. Identity retained. Direction of margin decrease is descriptive, not a task-isolation guarantee.',selection='Exactly14 pre-existing source-group evaluation pairs, all28endpoints. No competence/effect filtering, no training IDs. Keep every source alignment failure. No representation training or best-site selection.',stop='One fixed native coverage map across all families/layers/regions, independently audit. Use informative and null targets together for subsequent common-target featurizer design; do not tune per-family layers based on outcomes.')
 out['design_sha256']=sha(V/'coverage_source_holdout_design.json'); out['scope']='Familiar source-group development transfer, not fresh whole-task transfer'; out['families_without_evaluation']=sorted(({r['family'] for r in panel['rows']}|{'object_color'})-{r['family'] for r in pairs}); op=V/'source_holdout_native_protocol.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(pairs),'pairs',len(assignments),'assignments',sum(x['eligible'] for x in assignments),'eligible')
if __name__=='__main__':main()
