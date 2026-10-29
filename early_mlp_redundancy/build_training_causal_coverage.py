"""M57 freeze training-only early-MLP causal coverage without outcomes."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 panel=json.loads((V/'training_source_context_audit.json').read_text());split=json.loads((V/'diverse_supervision_split.json').read_text());train=set(split['train_pair_ids']);pairs=[]
 for p in panel['rows']:
  if p['id'] in train:pairs.append({k:p[k] for k in ['id','family','token_ids','answer_groups','variants']}|dict(source='old_training_panel'))
 assert len(pairs)==66
 color=json.loads((V/'broader_training_screen_manifest.json').read_text())
 for p in color['pairs']:
  if p['family']=='object_color' and p['format']=='few_shot_completion':pairs.append(dict(id=p['id'],family=p['family'],token_ids=[e['ids'] for e in p['endpoints']],answer_groups=[e['answer_groups'] for e in p['endpoints']],variants=[0,1],source='M55_training_screen'))
 exclusions={'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense','continent_recall','object_function'};assert not exclusions&{p['family'] for p in pairs};assert not set(split['evaluation_pair_ids'])&{p['id'] for p in pairs};assignments=[]
 for p in pairs:
  a,b=p['token_ids'];equal=len(a)==len(b);changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y] if equal else [];p.update(equal_length=equal,changed_positions=changed,source_region_eligible=equal and bool(changed),source_ineligibility_reason=None if equal and changed else 'unequal_length' if not equal else 'identical_token_sequence')
  for ep in [0,1]:
   for layer in [0,1,2]:
    for region in ['changed_input_positions','final_position','identity']:
     eligible=region!='changed_input_positions' or p['source_region_eligible'];positions=changed if region=='changed_input_positions' else [len(p['token_ids'][ep])-1] if region=='final_position' else [];assignments.append(dict(pair_id=p['id'],endpoint=ep,layer=layer,region=region,eligible=eligible,recipient_positions=positions,donor_positions=changed if region=='changed_input_positions' else [len(p['token_ids'][1-ep])-1] if region=='final_position' else []))
 out=dict(stage='frozen_before_native_outcomes',source_sha256=sha(Path(__file__)),input_hashes={n:sha(V/n) for n in ['training_source_context_audit.json','diverse_supervision_split.json','broader_training_screen_manifest.json']},pairs=pairs,assignments=assignments,layers=[0,1,2],training_exclusions=sorted(exclusions),semantics='Replace native product write at selected positions with opposite endpoint write at aligned addresses, preserving Down bias. Changed-token positions use index equality only for equal-length prompts: this is an address-based contrast, not semantic alignment. Final positions align explicitly even with unequal lengths. Donors may change multiple linguistic variables. Intervene one layer/region at a time; no simultaneous layer patch.',metrics='Final full-vocabulary native-weighted squared logprob response, output KL, intended-vs-alternate margin change, task competence and raw delta energy. No norm matching or amplitudes. Identity retained. Direction of margin decrease is descriptive, not a task-isolation guarantee.',selection='All66 existing training pairs plus4color pairs. No competence/effect filtering or evaluation pairs. Keep every source alignment failure. No representation training or best-site selection.',stop='One fixed native coverage map across all families/layers/regions, independently audit. Use informative and null targets together for subsequent common-target featurizer design; do not tune per-family layers based on outcomes.')
 op=V/'training_causal_coverage_protocol.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(pairs),'pairs',len(assignments),'assignments',sum(x['eligible'] for x in assignments),'eligible')
if __name__=='__main__':main()
