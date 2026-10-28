"""M30 fixed task-span deltas, paired training and evaluation wrappers."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
up=P/'unseen_direction_protocol.json';u=json.loads(up.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());ep=P/'context_expansion_protocol.json';expanded=json.loads(ep.read_text());byid={x['id']:x for x in expanded['rows']};wrappers=sorted({x['template'] for x in expanded['rows'] if x['template_split']=='evaluation'});assert len(wrappers)==2;rows=[]
for spec in u['rows']:
 if spec['region']!='task':continue
 i=spec['target_index'];old=t['records'][i];src=byid[old['pair_id']];endpoint=old['endpoint'];assert src['lexical_split']=='train';task=src['token_ids'][endpoint][src['prefix_length']:]
 for context,pairid in [('reference',src['id'])]+[(w,w+'|'+src['base_id']) for w in wrappers]:
  dst=byid[pairid];assert dst['base_id']==src['base_id'] and dst['lexical_split']=='train';assert dst['token_ids'][endpoint][dst['prefix_length']:]==task;assert dst['answer_groups'][endpoint]==src['answer_groups'][endpoint] and dst['variants'][endpoint]==src['variants'][endpoint];assert context=='reference' or dst['template_split']=='evaluation';rows.append(dict(target_index=i,kind=spec['kind'],seed=spec['seed'],family=spec['family'],group=spec['group'],context=context,source_pair_id=src['id'],pair_id=dst['id'],endpoint=endpoint,source_prefix_length=src['prefix_length'],prefix_length=dst['prefix_length'],task_token_ids=task))
assert len(rows)==720
out=dict(unseen_protocol_sha256=sha(up),targets_sha256=sha(tp),expansion_sha256=sha(ep),models=u['models'],rows=rows,contexts=['reference']+wrappers,scope='All240 M28 task-region scheduled rows crossed with reference and both previously excluded training templates. Exact task tokens and original M18 reference task-span delta fixed; no prefix intervention. Retain repeats/noops/native failures/all families/all20 candidates. Familiar development inputs; not pristine holdout or prefix-failure test.',stopping='One paired crossover and complete audit; no model fits, effect-based sampling or candidate selection.')
f=P/'wrapper_crossover_protocol.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(rows),out['contexts'])
