"""M115 token/label/logit/edge audit and predeclared competence gates."""
import json
import numpy as np
import torch
import tiktoken
from fit_pca_encoder import V,sha


def main():
 torch.set_num_threads(1);pp=V/'sentiment_composition_protocol.json';p=json.loads(pp.read_text());rp=V/'sentiment_composition_native.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(V.parent/'probe_sentiment_composition.py');assert sha(r['path'])==r['sha256'];saved=torch.load(r['path'],weights_only=True);tok=tiktoken.get_encoding('gpt2');observed={x['id']:x for x in r['rows']};items={x['id']:x for x in p['rows']};maxerr=0.
 for item in p['rows']:
  assert tok.encode(item['prompt'])==item['token_ids'] and saved[item['id']]['ids'][0].tolist()==item['token_ids'];z=saved[item['id']]['logits'].double().numpy();scores=[]
  for group in item['answer_groups']:
   zz=z[group];m=zz.max();scores.append(m+np.log(np.exp(zz-m).sum()))
  out=observed[item['id']];choice=int(np.argmax(scores));top=int(z.argmax());assert choice==out['chosen_group'] and top==out['top1_token'];assert (choice==item['expected_group'])==out['group_correct'];assert (top in item['answer_groups'][item['expected_group']])==out['top1_correct'];err=abs(scores[0]-scores[1]-out['positive_minus_negative_margin']);maxerr=max(maxerr,err);assert err<1e-10
  if item['cohort']=='factorial':
   words=p['lexical_pairs'][item['lexicon']];word=words[item['valence']];modifier='not' if item['negated'] else 'really';assert f'was {modifier} {word}.' in item['prompt'];expected=item['valence'] if modifier=='really' else 1-item['valence'];assert expected==item['expected_group']
 for edge in p['edges']:
  a,b=items[edge['first']],items[edge['second']];factor=edge['factor'];other='negated' if factor=='valence' else 'valence';assert a[factor]==0 and b[factor]==1 and a[other]==b[other]
  assert all(a[k]==b[k] for k in ['format','lexicon','subject']);assert len(a['token_ids'])==len(b['token_ids']);assert [i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y]==edge['changed_positions']
 assert len(saved)==len(items)==len(observed)==160 and len(p['edges'])==128 and r['forwards']==160 and r['exact_historical_replays']==32
 cells=[];edges=[];formats=[]
 for fmt in ['review','text']:
  for valence in [0,1]:
   for negated in [0,1]:
    rr=[x for x in r['rows'] if x.get('format')==fmt and x['valence']==valence and x['negated']==negated];assert len(rr)==16
    cells.append(dict(format=fmt,valence=valence,negated=negated,n=16,group_correct=sum(x['group_correct'] for x in rr),top1_correct=sum(x['top1_correct'] for x in rr),mean_positive_margin=float(np.mean([x['positive_minus_negative_margin'] for x in rr]))))
  for factor in ['valence','negated']:
   other='negated' if factor=='valence' else 'valence'
   for setting in [0,1]:
    ee=[e for e in p['edges'] if e['factor']==factor and items[e['first']]['format']==fmt and items[e['first']][other]==setting];assert len(ee)==16
    edges.append(dict(format=fmt,factor=factor,other_setting=setting,n=16,both_correct=sum(observed[e['first']]['group_correct'] and observed[e['second']]['group_correct'] for e in ee),choice_flips=sum(observed[e['first']]['chosen_group']!=observed[e['second']]['chosen_group'] for e in ee)))
  cc=[x for x in cells if x['format']==fmt];ee=[x for x in edges if x['format']==fmt];passed=all(x['group_correct']>=12 and x['top1_correct']>=8 for x in cc) and all(x['both_correct']>=12 for x in ee)
  formats.append(dict(format=fmt,pass_gate=passed,group_correct=sum(x['group_correct'] for x in cc),top1_correct=sum(x['top1_correct'] for x in cc),n=64))
 result=dict(status='passed',source_sha256=sha(__file__),native_sha256=sha(rp),audit=dict(rows=160,edges=128,max_margin_discrepancy=maxerr),cells=cells,edges=edges,formats=formats,historical=dict(n=32,group_correct=sum(x['group_correct'] for x in r['rows'] if x['cohort']=='historical_replay'),top1_correct=sum(x['top1_correct'] for x in r['rows'] if x['cohort']=='historical_replay')),scope='Predeclared semantic contrast and full-vocabulary gates. Binary not-polarity convention is a diagnostic, not an exhaustive semantics of negation. No semantic supervision licensed for failed formats.')
 dest=V/'sentiment_composition_audit.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
