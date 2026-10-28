"""M10 metadata-only split: no changed source token crosses task prompt splits."""
import hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ip=P/'validation/training_source_context_audit.json';panel=json.loads(ip.read_text());rows=panel['rows'];sources=[];prompts=[]
 for x in rows:
  source=x['source_token_pair'] or [x['token_ids'][ep][pos] for ep in [0,1] for pos in x['changed_positions'] if pos<len(x['token_ids'][ep])];assert source;sources.append(set(source));prompts.append(set(sum(x['token_ids'],[])))
 def components(strict):
  parents=list(range(len(rows)))
  def root(i):
   while parents[i]!=i:i=parents[i]
   return i
  for i in range(len(rows)):
   for j in range(i):
    if (sources[i]&prompts[j] or sources[j]&prompts[i]) if strict else sources[i]&sources[j]:parents[root(i)]=root(j)
  groups={}
  for i in range(len(rows)):groups.setdefault(root(i),[]).append(i)
  return sorted(groups.values(),key=lambda g:min(rows[i]['id'] for i in g))
 weak=components(False);groups=components(True);families=list(panel['families']);familygroups={f:[j for j,g in enumerate(groups) if any(rows[i]['family']==f for i in g)] for f in families};single=[f for f in families if len(familygroups[f])==1];forced={familygroups[f][0] for f in single};rng=random.Random(2026102810)
 # Search uses only incidence metadata, never logits, losses or model outcomes.
 for attempt in range(10000):
  train_groups={j for j in range(len(groups)) if j in forced or rng.random()<.5}
  if all(bool(set(gs)&train_groups) and (len(gs)==1 or bool(set(gs)-train_groups)) for gs in familygroups.values()):break
 else:raise RuntimeError('No split satisfying metadata coverage constraints')
 train=sorted(i for j,g in enumerate(groups) if j in train_groups for i in g);evaluation=sorted(set(range(len(rows)))-set(train));trainprompt=set().union(*(prompts[i] for i in train));evalprompt=set().union(*(prompts[i] for i in evaluation));trainsource=set().union(*(sources[i] for i in train));evalsource=set().union(*(sources[i] for i in evaluation));assert not trainsource&evalprompt and not evalsource&trainprompt
 report=[]
 for f in families:
  tr=[i for i in train if rows[i]['family']==f];ev=[i for i in evaluation if rows[i]['family']==f];report.append(dict(family=f,strict_components=len(familygroups[f]),train_pairs=len(tr),evaluation_pairs=len(ev),claim='No lexical transfer evaluation possible; training-only' if not ev else 'Cross-source-input-token evaluation within previously examined development family'))
 out=dict(status='frozen_metadata_split',panel_sha256=sha(ip),source_sha256=sha(Path(__file__)),selection_seed=2026102810,metadata_only_proposals_checked=attempt+1,weak_changed_token_components=len(weak),strict_source_to_prompt_components=len(groups),groups=[dict(group=j,pair_ids=[rows[i]['id'] for i in g],source_tokens=sorted(set().union(*(sources[i] for i in g))),split='train' if j in train_groups else 'evaluation') for j,g in enumerate(groups)],train_pair_ids=[rows[i]['id'] for i in train],evaluation_pair_ids=[rows[i]['id'] for i in evaluation],family_summary=report,scope='No changed source token from either split occurs anywhere in the other split task prompts, exact tokenizer IDs. Ordinary context tokens may overlap; semantic synonyms/subwords/corpus vocabulary are not held out. All examples historically inspected. Teachers produce distributions over full vocabulary, so this is input-token separation, NOT unseen output labels, novel words to model, pristine validation or new-task discovery. Single-component families deliberately training-only. No original excluded task family read.')
 op=P/'validation/diverse_supervision_split.json';op.write_text(json.dumps(out,indent=2)+'\n')
 m8p=P/'validation/causal_featurizers.json';m8=json.loads(m8p.read_text());initial=[x for x in m8['models'] if x['objective']=='downstream_KL'];schedules={}
 for seed in [71,72]:
  rr=random.Random(10000+seed);natural=[rr.randrange(32) for _ in range(120)];taskfamilies=[]
  for epoch in range(6):order=families.copy();rr.shuffle(order);taskfamilies.extend(order)
  taskpairs=[rr.choice([rows[i]['id'] for i in train if rows[i]['family']==f]) for f in taskfamilies];schedules[seed]=dict(natural_documents=natural,mixed_task_pairs=taskpairs)
 protocol=dict(status='frozen_before_training',split_sha256=sha(op),M8_sha256=sha(m8p),initial_models=initial,arms=['natural_only_KL','mixed_natural_task_KL'],steps=120,learning_rate=.0003,optimizer='fresh Adam; unit decoder columns after each update',schedules=schedules,design='Four M8causal checkpoints scalar/block×seed71/72; two arms each. Pure natural uses listed trainingdocument at all120updates. Mixed uses SAME natural doc on even0-based steps and listed taskpair on oddsteps. Six taskupdates per each10families, pair loss equal mean of endpoint KL, each endpoint mean overitspositions/all50304logits. No binary/answer label loss. All inputs alreadycomputed nativeMLP1writes. Matchedupdates/initialweights; notmatchedtokens,FLOPs orpureadditionaldata intervention. Naturaltrainingdocs0–31 only.',evaluation='Fixedfinalstep. Report source-separated evaluationpair metrics for6eligiblefamilies, trainingpair performance separately(all10families), and unchanged naturaldevelopmentdocs32–63 forforgetting. Four training-onlyfamilies have NO lexical evaluation estimate. Compare frozenM8baseline andbotharms; no eval-based selection or tuning.',limits='Prospective split/training protocol within previously examined development data, not pristine confirmation. Exact source-inputtoken separation doesnotexclude synonyms/semantic overlap or prior natural-text exposure. Original whole-task exclusions stayreserved. No semanticmanifold/discoveredtaskclaim.')
 (P/'validation/diverse_supervision_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('TRAIN',len(train),'EVALUATION',len(evaluation),'ATTEMPTS',attempt+1);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
