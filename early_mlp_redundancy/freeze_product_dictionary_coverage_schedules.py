"""M98 metadata-only matched family sequences and coverage schedules."""
import json,random
from collections import Counter
from fit_pca_encoder import V,sha

def main():
 inv=json.loads((V/'product_dictionary_coverage_inventory.json').read_text());assert inv['status']=='coverage_measured';pairs={p['id']:p for p in json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs']};old=json.loads((V/'balanced_response_training_protocol.json').read_text())['schedules'];schedules={}
 for seed in [71,72]:
  ids=sorted(pairs);random.Random(98000+seed).shuffle(ids);family_counts=Counter(p['family'] for p in pairs.values());visited={x['pair_id'] for x in old[str(seed)]['task_updates']};rng=random.Random(98100+seed);control={}
  for family,n in sorted(family_counts.items()):
   subset=sorted(pid for pid in visited if pairs[pid]['family']==family);assert subset and len(subset)<=n;draws=subset+[rng.choice(subset) for _ in range(n-len(subset))];rng.shuffle(draws);control[family]=draws
  full=[dict(pair_id=pid,family=pairs[pid]['family']) for pid in ids];resampled=[dict(pair_id=control[x['family']].pop(),family=x['family']) for x in full];assert all(not x for x in control.values());assert {x['pair_id'] for x in resampled}==visited
  natural=old[str(seed)]['natural_documents']+old[str(seed)]['natural_documents'][:4];assert len(natural)==70
  schedules[str(seed)]={'full_pool':dict(task_updates=full,natural_documents=natural),'old_visited':dict(task_updates=resampled,natural_documents=natural)}
 out=dict(status='frozen_before_fitting',source_sha256=sha(__file__),design_sha256=sha(V/'product_dictionary_coverage_design.json'),inventory_sha256=sha(V/'product_dictionary_coverage_inventory.json'),schedules=schedules,randomization=dict(full_pool_shuffle='Python random.Random(98000+seed), sorted pair IDs',old_control='Python random.Random(98100+seed), sorted families/IDs; every oldvisited pair once, remainingfamily slots sampled with replacement, then shuffled'),scope='Same70-family sequence and70-natural-document schedule in bothconditions; onlypaircoverage/repetition differs. Everyoldvisitedpair appearsatleastonce toavoid accidental loss ofoldcoverage. Task/sourceeligibility andtoken lengths maydiffer; update/endpointbudget matched, notexactFLOPs.')
 p=V/'product_dictionary_coverage_schedules.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
