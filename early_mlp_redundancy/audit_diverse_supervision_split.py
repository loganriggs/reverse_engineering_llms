"""Independent BFS component and schedule audit of M10 metadata split."""
import json,hashlib,collections
from pathlib import Path
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sp=P/'diverse_supervision_split.json';split=json.loads(sp.read_text());pp=P/'diverse_supervision_protocol.json';protocol=json.loads(pp.read_text());ip=P/'training_source_context_audit.json';panel=json.loads(ip.read_text());assert sha(ip)==split['panel_sha256'];assert sha(sp)==protocol['split_sha256'];rows={r['id']:r for r in panel['rows']};sources={};alltokens={}
for key,row in rows.items():
 a,b=row['token_ids'];assert len(a)==len(b), 'Explicit alignment required for unequal lengths';changed=[j for j,(x,y) in enumerate(zip(a,b)) if x!=y];assert changed==row['changed_positions'];sources[key]={token for j in changed for token in [a[j],b[j]]};alltokens[key]=set(a+b)
adj={key:set() for key in rows}
for a in rows:
 for b in rows:
  if a!=b and (sources[a]&alltokens[b] or sources[b]&alltokens[a]):adj[a].add(b)
remaining=set(rows);groups=[]
while remaining:
 start=min(remaining);queue=[start];group=set()
 while queue:
  key=queue.pop()
  if key in group:continue
  group.add(key);queue.extend(adj[key]-group)
 remaining-=group;groups.append(group)
assert {frozenset(g) for g in groups}=={frozenset(g['pair_ids']) for g in split['groups']}
train=set(split['train_pair_ids']);evaluation=set(split['evaluation_pair_ids']);assert not train&evaluation and train|evaluation==set(rows)
for a in train:
 for b in evaluation:assert not sources[a]&alltokens[b] and not sources[b]&alltokens[a]
for row in split['family_summary']:
 f=row['family'];assert sum(rows[i]['family']==f for i in train)==row['train_pairs'];assert sum(rows[i]['family']==f for i in evaluation)==row['evaluation_pairs']
for stream in protocol['schedules'].values():
 assert len(stream['natural_documents'])==120 and all(0<=d<32 for d in stream['natural_documents']);assert len(stream['mixed_task_pairs'])==60;assert set(stream['mixed_task_pairs'])<=train;counts=collections.Counter(rows[i]['family'] for i in stream['mixed_task_pairs']);assert len(counts)==10 and set(counts.values())=={6}
for model in protocol['initial_models']:assert sha(Path(model['path']))==model['sha256'] and model['objective']=='downstream_KL'
out=dict(status='passed',independent_BFS_components=len(groups),train_pairs=len(train),evaluation_pairs=len(evaluation),source_tokens_rebuilt_from_endpoint_differences=True,source_to_other_split_prompt_intersections=0,training_families=10,evaluation_families=sum(r['evaluation_pairs']>0 for r in split['family_summary']),paired_schedules_and_frozen_model_hashes_checked=True,split_sha256=sha(sp),protocol_sha256=sha(pp),model_forwards=0)
(P/'diverse_supervision_split_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
