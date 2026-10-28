"""M17 metadata-only context expansion; fixed template and source-token splits."""
from pathlib import Path
import json,hashlib,tiktoken
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
TEMPLATES=[('consider','train','Consider the following text.\n'),('read','train','Read this passage.\n'),('text','train','Text:\n'),('exercise','train','For this exercise:\n'),('passage','evaluation','A passage begins below.\n'),('sample','evaluation','Sample text follows:\n')]
def main():
 bp=P/'training_source_context_audit.json';sp=P/'diverse_supervision_split.json';b=json.loads(bp.read_text());s=json.loads(sp.read_text());assert sha(bp)==s['panel_sha256'];enc=tiktoken.get_encoding('gpt2');train=set(s['train_pair_ids']);evaluation=set(s['evaluation_pair_ids']);assert not train&evaluation;rows=[];source={split:set(t for x in b['rows'] if x['id'] in ids for ep in [0,1] for j in x['changed_positions'] for t in [x['token_ids'][ep][j]]) for split,ids in [('train',train),('evaluation',evaluation)]}
 for name,split,prefix in TEMPLATES:
  pre=enc.encode(prefix)
  for x in b['rows']:
   lexical='train' if x['id'] in train else 'evaluation';assert x['id'] in train|evaluation;texts=[prefix+t for t in x['texts']];ids=[enc.encode(t) for t in texts]
   for ep in [0,1]:
    assert ids[ep]==pre+x['token_ids'][ep], 'Boundary merge'
    assert not set(ids[ep])&source['evaluation' if lexical=='train' else 'train'], ('Cross-split source leak',name,x['id'])
   rows.append(dict(id=name+'|'+x['id'],base_id=x['id'],family=x['family'],template=name,template_split=split,lexical_split=lexical,training_eligible=split=='train' and lexical=='train',prefix_length=len(pre),texts=texts,token_ids=ids,answer_groups=x['answer_groups'],variants=x['variants']))
 assert len(rows)==480;assert len({t for r in rows for t in r['texts']})==960
 protocol=dict(status='frozen_metadata',source_sha256=sha(Path(__file__)),bank_sha256=sha(bp),lexical_split_sha256=sha(sp),templates=[dict(id=n,split=s,prefix=p) for n,s,p in TEMPLATES],rows=rows,competence_gate='Per template x lexical_split x family: >=75%pairs both endpoints binary correct AND >=50% endpoints native full-vocabulary top1 correct. All rows retained regardless gate. Training sampler may use only passing train-template/train-lexical cohorts; do not use evaluation competence to choose training rows. No replacement prompts or thresholds.',scope='Six new fixed wrappers on80 familiar development pairs/10 allowed families. Whole-template training exclusion, not semantic novelty or pristine evaluation. Original whole-task exclusions untouched. Four families have no lexical evaluation component; evaluate only context transfer for those.',future_comparison='Freeze four M11 mixed starts, original-context vs expanded-context task exposure with paired family/source schedules and matched update counts, same natural-text interleaving. Report natural fidelity and each lexical/template quadrant. Define native-vs-reconstructed intervention fidelity before fitting; retain both prefix/task regions. No group/gain search on existing14pair panel.',stopping_rule='One fixed context expansion/screen, no prompt replacement after outcomes. If training coverage is inadequate, report missing coverage before a separately specified change.')
 out=P/'context_expansion_protocol.json'
 if out.exists():assert json.loads(out.read_text())==protocol
 else:out.write_text(json.dumps(protocol,indent=2)+'\n')
 print('Frozen480pairs/960endpoints;264training-eligiblepairs;exactboundariesandsource-tokenexclusionpassed')
if __name__=='__main__':main()
