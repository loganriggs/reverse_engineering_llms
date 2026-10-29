"""M50 activation-blind same-token/same-group direction-transfer inventory."""
from pathlib import Path
import hashlib,json,collections
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pp=V/'shared_context_inventory_protocol.json';q=json.loads(pp.read_text());assert q['source_sha256']==sha(Path(__file__));ap=V/'shared_group_native_protocol.json';assert sha(ap)==q['assignment_protocol_sha256'];assign={x['key']:x['group'] for x in json.loads(ap.read_text())['assignments']};panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());train=set(json.loads((V/'diverse_supervision_split.json').read_text())['train_pair_ids']);inputs=[]
 for row in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{row['id']}:{ep}",domain='task',split='train' if row['id'] in train else 'evaluation',tokens=row['token_ids'][ep]))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',domain='natural',split='evaluation',tokens=cohort['token_ids'][doc]))
 units=[];lookup={};occurrences=[];groups=collections.defaultdict(list)
 for c in inputs:
  g=assign[c['key']]
  for pos,t in enumerate(c['tokens']):
   prefix=c['tokens'][:pos+1];key=c['domain'],c['split'],g,tuple(prefix)
   if key not in lookup:
    i=len(units);lookup[key]=i;units.append(dict(id=i,domain=c['domain'],split=c['split'],group=g,token=t,prefix=prefix,source_key=c['key'],position=pos));groups[c['domain'],c['split'],g,t].append(i)
   occurrences.append(dict(key=c['key'],position=pos,unit=lookup[key]))
 donors=list(range(len(units)));eligible=[False]*len(units)
 for key,ii in groups.items():
  if len(ii)<2:continue
  order=sorted(ii,key=lambda i:hashlib.sha256(json.dumps(units[i]['prefix'],separators=(',',':')).encode()).hexdigest())
  for j,i in enumerate(order):donors[i]=order[(j+1)%len(order)];eligible[i]=True
 assert sorted(donors)==list(range(len(units)))
 for i,j in enumerate(donors):
  assert all(units[i][k]==units[j][k] for k in ['domain','split','group','token'])
  assert (units[i]['prefix']!=units[j]['prefix'])==eligible[i]
 summaries=[]
 for domain,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
  ii=[u['id'] for u in units if (u['domain'],u['split'])==(domain,part)];occ=[x for x in occurrences if x['unit'] in set(ii)];ee=[i for i in ii if eligible[i]];summaries.append(dict(domain=domain,split=part,unique_prefix_group_units=len(ii),eligible_units=len(ee),unpaired_units=len(ii)-len(ee),occurrences=len(occ),eligible_occurrences=sum(eligible[x['unit']] for x in occ),eligible_token_types=len({units[i]['token'] for i in ee}),eligible_groups=len({units[i]['group'] for i in ee})))
 # Independent reconstruction from occurrence records verifies every prefix/group exactly.
 expected={c['key']:c for c in inputs};assert len(occurrences)==sum(len(c['tokens']) for c in inputs);seen=set()
 for o in occurrences:
  c=expected[o['key']];u=units[o['unit']];assert u['prefix']==c['tokens'][:o['position']+1] and u['group']==assign[o['key']] and u['domain']==c['domain'] and u['split']==c['split'];seen.add((o['key'],o['position']))
 assert len(seen)==len(occurrences)
 out=dict(status='passed',protocol_sha256=sha(pp),source_hashes={str(f.name):sha(f) for f in [ap,V/'training_source_context_audit.json',V/'natural_token_context_cohort.json',V/'diverse_supervision_split.json']},units=units,occurrences=occurrences,donors=donors,eligible=eligible,summaries=summaries,scope='Activation/outcome-blind inventory only. Same token/group/domain/split, distinct exact causal prefixes; no eligibility on norm, effect or support. No native direction/output transfer measured.')
 f=V/'shared_context_inventory.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
