"""M115 fixed sentiment-composition diagnostic; entire panel excluded from fitting."""
import json,re
import tiktoken
from fit_pca_encoder import V,A,sha


def main():
 tok=tiktoken.get_encoding('gpt2');tp=V/'training_causal_coverage_protocol.json';train=json.loads(tp.read_text())['pairs'];sent=[x for x in train if x['family']=='sentiment'];assert len(sent)==16
 alltrain=[tuple(ids) for pair in train for ids in pair['token_ids']];terms=[]
 for i in range(4):
  pair=next(x for x in sent if x['id']==f'old:sentiment-context0-{i}');texts=[tok.decode(ids) for ids in pair['token_ids']];terms.append([re.fullmatch(r'Review: The movie was (.+)\. Sentiment:',t).group(1) for t in texts])
 groups=sent[0]['answer_groups'][0];assert all(x['answer_groups']==[groups,groups] and x['variants']==[0,1] for x in sent)
 subjects=['movie','book','meal','game'];rows=[]
 for fmt in ['review','text']:
  for lex,words in enumerate(terms):
   for subject in subjects:
    for valence in [0,1]:
     for negated in [0,1]:
      modifier=['really','not'][negated];sentence=f'The {subject} was {modifier} {words[valence]}.';prompt=(f'Review: {sentence} Sentiment:' if fmt=='review' else f'Text: {sentence} Overall sentiment:');ids=tok.encode(prompt)
      for group in groups:
       for answer in group:assert tok.encode(prompt+tok.decode([answer]))==ids+[answer]
      rows.append(dict(id=f'm115:{fmt}:{lex}:{subject}:{valence}:{negated}',cohort='factorial',format=fmt,lexicon=lex,subject=subject,valence=valence,negated=negated,modifier=modifier,word=words[valence],prompt=prompt,token_ids=ids,answer_groups=groups,expected_group=valence^negated))
 assert len({tuple(x['token_ids']) for x in rows})==128
 overlap=sum(tuple(x['token_ids']) in alltrain for x in rows);assert overlap==0
 edges=[]
 for r in rows:
  for factor in ['valence','negated']:
   if r[factor]:continue
   other=next(x for x in rows if x['format']==r['format'] and x['lexicon']==r['lexicon'] and x['subject']==r['subject'] and x['valence']==(1 if factor=='valence' else r['valence']) and x['negated']==(1 if factor=='negated' else r['negated']))
   aligned=len(r['token_ids'])==len(other['token_ids']);pos=[i for i,(a,b) in enumerate(zip(r['token_ids'],other['token_ids'])) if a!=b] if aligned else None;edges.append(dict(factor=factor,first=r['id'],second=other['id'],equal_length=aligned,changed_positions=pos))
 for pair in sent:
  for ep in [0,1]:rows.append(dict(id=pair['id']+f'|{ep}',cohort='historical_replay',prompt=tok.decode(pair['token_ids'][ep]),token_ids=pair['token_ids'][ep],answer_groups=groups,expected_group=pair['variants'][ep]))
 oldexamples=json.loads((A/'bilin18_broad_task_competence.json').read_text())['examples']
 inventory=dict(training_protocol_sha256=sha(tp),training_family_counts={f:sum(x['family']==f for x in train) for f in sorted({x['family'] for x in train})},sentiment_training_prompts=[tok.decode(ids) for x in sent for ids in x['token_ids']],all_plain_sentiment=all(' not ' not in tok.decode(ids) and ' really ' not in tok.decode(ids) for x in sent for ids in x['token_ids']),historical_broad_screen_sha256=sha(A/'bilin18_broad_task_competence.json'),historical_semantic_redteam_sha256=sha(A/'bilin18_broad_task_competence_redteam.json'),whole_task_exclusion_sha256=sha(V/'whole_task_native_protocol.json'))
 p=dict(status='frozen_before_new_outputs',source_sha256=sha(__file__),inventory=inventory,lexical_pairs=terms,subjects=subjects,groups=groups,rows=rows,edges=edges,training_exact_prompt_overlap=overlap,scope='Diagnostic validation only; no featurizer training eligibility. Negation was studied historically; new factorial coverage is not new-task discovery. Original whole-task exclusions unchanged.',labels='Binary sentiment convention: really preserves adjective polarity; not reverses expected polarity. This is an operational sentiment contrast, not a claim that linguistic negation universally implements Boolean inversion.',gate='Separately for each format: every valence×modifier cell has>=75% semantic group-choice accuracy and>=50% semantic full-vocab top1; for each factor and each setting of the other factor,>=75% edges have both endpoints correct. Groups inherited unchanged from training panel. Report all cells and failed formats.',stop='One fixed panel. No prompt repair, answer-group expansion, threshold changes or post-result lexical selection. Competence precedes any intended semantic causal supervision.',reserved='All128factorial prompts excluded from fitting. Replays32exact historical controls; no new family added to training.')
 dest=V/'sentiment_composition_protocol.json';assert not dest.exists();dest.write_text(json.dumps(p,indent=2)+'\n');print('FROZEN','TERMS',terms,'ROWS',len(rows),'EDGES',len(edges),'ALIGNED',sum(x['equal_length'] for x in edges))
if __name__=='__main__':main()
