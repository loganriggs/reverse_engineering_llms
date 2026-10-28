"""M18 independent protocol/schedule/selection audit, before fitting."""
from pathlib import Path
from collections import Counter
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(tag,x):return hashlib.sha256((tag+'|'+x).encode()).hexdigest()
pp=P/'context_training_protocol.json';p=json.loads(pp.read_text())
for field,name in [('expansion_sha256','context_expansion_protocol.json'),('competence_sha256','context_expansion_competence.json'),('split_sha256','diverse_supervision_split.json'),('M11_results_sha256','diverse_supervision_results.json')]:assert p[field]==sha(P/name)
e=json.loads((P/'context_expansion_protocol.json').read_text());c=json.loads((P/'context_expansion_competence.json').read_text());s=json.loads((P/'diverse_supervision_split.json').read_text());bank={x['id']:x for x in json.loads((P/'training_source_context_audit.json').read_text())['rows']};rows={x['id']:x for x in e['rows']};gates={(x['template'],x['lexical_split'],x['family']):x['pass_gate'] for x in c['cohorts']};coverage=[]
assert len(p['initial_models'])==4 and {(x['kind'],x['seed']) for x in p['initial_models']}=={(k,s) for k in ['scalar','block'] for s in [71,72]}
for x in p['initial_models']:assert sha(Path(x['path']))==x['sha256']
assert p['arms']==['original_context','expanded_context'];assert p['steps']==120 and p['learning_rate']==.0003
for seed,schedule in p['schedules'].items():
 assert len(schedule['natural_documents'])==120 and all(0<=d<32 for d in schedule['natural_documents']);assert len(schedule['task_updates'])==60;families=Counter();templates={}
 for i,u in enumerate(schedule['task_updates']):
  assert u['task_update']==i;row=rows[u['expanded_pair_id']];assert row['base_id']==u['source_pair_id'];assert row['training_eligible'];assert row['lexical_split']==row['template_split']=='train';assert row['base_id'] in s['train_pair_ids'];assert gates[row['template'],row['lexical_split'],row['family']];assert row['family']==u['family']==bank[row['base_id']]['family'];assert row['answer_groups']==bank[row['base_id']]['answer_groups'];assert row['variants']==bank[row['base_id']]['variants'];families[u['family']]+=1;templates.setdefault(u['family'],set()).add(row['template'])
 assert len(families)==10 and set(families.values())=={6};coverage.append(dict(seed=int(seed),family_updates=dict(families),templates_by_family={f:sorted(t) for f,t in templates.items()}))
expected=[]
for ls,ts in [('evaluation','train'),('train','evaluation'),('evaluation','evaluation')]:
 candidates=[x for x in rows.values() if x['lexical_split']==ls and x['template_split']==ts and gates[x['template'],ls,x['family']]]
 for family in sorted({x['family'] for x in candidates}):
  chosen=min([x for x in candidates if x['family']==family],key=lambda x:rank('causal-pair-v1',x['id']));expected.append(dict(pair_id=chosen['id'],family=family,lexical_split=ls,template_split=ts))
assert expected==p['causal_pairs'] and len(expected)==22
seen=set()
for r in p['causal_interventions']:
 key=(r['pair_id'],r['endpoint'],r['seed'],r['region']);assert key not in seen;seen.add(key);assert r['region'] in ['prefix','task'] and r['endpoint'] in [0,1] and r['seed'] in [71,72];assert r['pair_id'] in {x['pair_id'] for x in expected};assert r['groups']==sorted(range(32),key=lambda g:rank(f'causal-group-v1:{r["pair_id"]}:{r["endpoint"]}:{r["seed"]}',str(g)))[:2]
assert len(seen)==22*2*2*2
out=dict(status='passed_before_training',protocol_sha256=sha(pp),models=4,fits=8,updates_per_fit=120,task_updates_per_family=6,causal_pairs=22,causal_endpoints=44,native_fixed_perturbations=704,training_coverage=coverage,scope='All input/model hashes, schedule pairing/eligibility, exact labels, no excluded templates in fitting, deterministic native-gated causal selection. Does not validate future optimizer/autograd or intervention implementation.')
(P/'context_training_protocol_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
