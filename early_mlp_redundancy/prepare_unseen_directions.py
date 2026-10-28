"""M28 freeze new groups by exact input identity, never by measured effect."""
from pathlib import Path
import hashlib,json,collections
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tp=P/'response_training_targets.json';t=json.loads(tp.read_text());cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];c=torch.load(cp,weights_only=True);identity={k:hashlib.sha256(json.dumps(v['ids'].tolist(),separators=(',',':')).encode()).hexdigest() for k,v in c['inputs'].items()};seen=collections.defaultdict(set)
for r in t['records']:seen[r['kind'],r['seed'],identity[r['input_key']]].add(r['group'])
rows=[]
for i,r in enumerate(t['records']):
 tokenhash=identity[r['input_key']];key=r['kind'],r['seed'],tokenhash;forbidden=sorted(seen[key]);available=set(range(32))-set(forbidden);assert available
 def rank(g):return hashlib.sha256(f"M28-unseen-v1|{r['kind']}|{r['seed']}|{tokenhash}|{r['region']}|{g}".encode()).hexdigest()
 group=min(available,key=rank);rows.append(dict(target_index=i,kind=r['kind'],seed=r['seed'],family=r['family'],input_key=r['input_key'],region=r['region'],token_hash=tokenhash,supervised_groups=forbidden,group=group))
models=[]
for name in ['finite_candidate_models','aggregate_kl_models']:
 rp=P/(name+'.json');r=json.loads(rp.read_text());assert json.loads((P/(name+'_audit.json')).read_text())['result_sha256']==sha(rp)
 for e in r['models']:
  if e['arm'].startswith('aggregate_') or (e['arm']=='initial' and name=='finite_candidate_models'):models.append(e)
assert len(models)==20
out=dict(targets_sha256=sha(tp),models=models,rows=rows,scope='Same scheduled training endpoints/regions and original M18 basis. Choose hash-min group among groups never supervised on EXACT token-id input for that kind/seed, across both regions and all aliases/repeats. No effect-based/active-group selection; retain inactive directions, repeats and all ten families. Discrete group exclusion is not geometric orthogonality or a whole-task holdout.',selection_salt='M28-unseen-v1',stopping='One held-direction diagnostic, all four starts and both fixed scales/objectives; no fitting or selection of candidates.')
op=P/'unseen_direction_protocol.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(rows),'models',len(models),'exact_inputs',len(set(identity.values())))
