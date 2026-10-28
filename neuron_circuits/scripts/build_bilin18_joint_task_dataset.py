"""Merge qualified task banks without mixing training and development holdouts."""
from pathlib import Path
import torch,json,hashlib
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts/bilin18_tensor_decomposition'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 specs=[('base','bilin18_expanded_task_bank','bilin18_task_bank_causal_sites_captures.pt',['agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender','sequence_copy','spatial_inverse']),('broader','bilin18_broader_task_bank','bilin18_broader_task_bank_captures.pt',['antonym','country_language'])]
 examples=[];pairs=[];vectors=[];all_vectors=[];hashes={str(Path(__file__)):sha(Path(__file__))}
 for bank,stem,capture,families in specs:
  p=A/f'{stem}.json';c=A/capture;hashes[str(p)]=sha(p);hashes[str(c)]=sha(c);screen=json.loads(p.read_text());writes=torch.load(c,weights_only=False,map_location='cpu')['writes'];gates={(r['family'],r['split']):r['pass_gate'] for r in screen['summary']}
  for row in screen['records']:
   if row['family'] not in families:continue
   assert all(i in writes for i in row['indices'])
   ii=[]
   for endpoint,source_i in enumerate(row['indices']):
    ii.append(len(examples));vectors.append(writes[source_i][12]);all_vectors.append(writes[source_i]);examples.append(dict(bank=bank,source_index=source_i,pair=row['id'],endpoint=endpoint,text=row['prompts'][endpoint],family=row['family'],split=row['split'],whole_task_holdout=row['whole_task_holdout'],native_pass=gates[row['family'],row['split']],training=row['split']=='discovery' and not row['whole_task_holdout']))
   pairs.append(dict(bank=bank,id=row['id'],indices=ii,family=row['family'],split=row['split'],training=examples[ii[0]]['training']))
 train=[i for i,r in enumerate(examples) if r['training']];assert len(train)==64;assert all(examples[i]['native_pass'] and not examples[i]['whole_task_holdout'] for i in train)
 assert {examples[i]['family'] for i in train}=={'agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender','antonym','country_language'}
 assert not any(examples[i]['family'] in ['sequence_copy','spatial_inverse','role_binding','tense','continent_recall','object_function'] for i in train)
 train_text={examples[i]['text'] for i in train}
 for r in examples:r['overlaps_training_prompt']=not r['training'] and r['text'] in train_text
 for pair in pairs:pair['clean_validation']=not pair['training'] and not any(examples[i]['overlaps_training_prompt'] for i in pair['indices'])
 unique_train=len({examples[i]['text'] for i in train});print('UNIQUE training prompts',unique_train)
 manifest=dict(training_unique_prompts=unique_train,training_duplicate_policy='Retain task-balanced slots; shared prompts are explicitly counted once for unique coverage. No claim of64 independent prompts.',layer=12,examples=examples,pairs=pairs,training_indices=train,source_hashes=hashes,limits='64-state seed dataset, not sufficient coverage; train selection uses discovery only. Native-failing validation cohorts retained with explicit flags. Broader contexts remain to be collected. No fitting/centering performed.')
 torch.save(dict(writes=torch.stack(vectors),writes_all_layers=torch.stack(all_vectors),manifest=manifest),A/'bilin18_joint_task_dataset.pt');(A/'bilin18_joint_task_dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('MERGED',len(examples),'states',len(train),'training',len(pairs),'pairs')
if __name__=='__main__':main()
