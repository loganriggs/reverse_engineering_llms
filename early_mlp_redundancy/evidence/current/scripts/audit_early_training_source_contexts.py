from pathlib import Path
import torch,json,hashlib,collections
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');P=Path('/home/loganriggs/Coding/reverse_engineering_llms/early_mlp_redundancy')
panelpath=A/'training_coverage_selector_comparison/panel.json';selection=json.loads(panelpath.read_text())['train']
packs={p:torch.load(A/d/'states.pt',map_location='cpu',weights_only=False,mmap=True) for p,d in [('old','multitask_boundary_contract'),('added','diverse_task_expansion_screen')]}
oldmeta={r['id']:r for r in json.loads((A/'multitask_boundary_contract/panel.json').read_text())}
elig=json.loads((A/'diverse_task_expansion_screen/training_eligibility.json').read_text())
rows=[];byfamily=collections.defaultdict(list)
for row in selection:
 ex=[]
 for key in row['indices']:
  bank,idx=key.split(':');ex.append(packs[bank]['examples'][int(idx)])
 assert row['family'] not in ['sequence_copy','spatial_inverse','weekday_successor','role_binding','tense']
 if row['id'].startswith('added:'):assert ex[0]['id'] in elig['pair_ids']
 a,b=[e['ids'] for e in ex];equal=len(a)==len(b);diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y] if equal else []
 one=equal and len(diff)==1;pos=diff[0] if one else None
 record=dict(id=row['id'],family=row['family'],original_indices=row['indices'],texts=[e['text'] for e in ex],token_ids=[a,b],answer_groups=[e['groups'] for e in ex],variants=[e['variant'] for e in ex],
  equal_length=equal,changed_positions=diff,single_token_edit=one,source_position=pos,
  source_prefix_before=a[:pos] if one else None,source_prefixes_through=[a[:pos+1],b[:pos+1]] if one else None,
  source_token_pair=[a[pos],b[pos]] if one else None,scope='Previously opened training-only pair, not fresh validation')
 rows.append(record);byfamily[row['family']].append(record)
summary={}
for family,rr in byfamily.items():
 single=[r for r in rr if r['single_token_edit']];groups=collections.defaultdict(set)
 for r in single:groups[tuple(r['source_token_pair'])].add(tuple(r['source_prefix_before']))
 summary[family]=dict(pairs=len(rr),single_token_pairs=len(single),unique_pre_edit_prefixes=len({tuple(r['source_prefix_before']) for r in single}),
  unique_directional_source_contexts=len({tuple(pre) for r in single for pre in r['source_prefixes_through']}),unique_token_pairs=len(groups),token_pairs_with_multiple_pre_edit_prefixes=sum(len(v)>1 for v in groups.values()),
  pre_edit_text_examples=[r['texts'][0] for r in single[:2]])
files=[panelpath,A/'multitask_boundary_contract/panel.json',A/'diverse_task_expansion_screen/training_eligibility.json']
result=dict(status='completed_metadata_audit_no_new_forwards',selection='Exact 80 historical training pairs; no evaluation examples exported or evaluated.',source_metadata_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},families=summary,rows=rows)
(P/'validation/training_source_context_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
