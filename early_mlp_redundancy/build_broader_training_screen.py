"""M55 fixed authored training-only formats; no model inference."""
from pathlib import Path
import json,hashlib
import tiktoken
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 design=json.loads((V/'broader_training_screen_design.json').read_text());tok=tiktoken.get_encoding('gpt2')
 specs={
 'comparative':dict(query=[('big','bigger'),('small','smaller'),('tall','taller'),('short','shorter'),('old','older'),('young','younger'),('fast','faster'),('slow','slower')],demo=[('warm','warmer'),('cool','cooler')],question='What is the comparative form of {x}?',instruction='Convert each adjective to its comparative form.'),
 'animal_sound':dict(query=[('dog','bark'),('cat','meow'),('cow','moo'),('sheep','baa'),('duck','quack'),('lion','roar'),('snake','hiss'),('bee','buzz')],demo=[('horse','neigh'),('pig','oink')],question='What sound does a {x} make?',instruction='Give the conventional written sound associated with each animal.'),
 'object_color':dict(query=[('snow','white'),('coal','black'),('grass','green'),('blood','red'),('a banana','yellow'),('an orange','orange'),('the sky','blue'),('milk','white')],demo=[('a ripe tomato','red'),('roasted coffee beans','brown')],question='What is the usual color of {x}?',instruction='Give the usual color of each object.'),
 'addition':dict(query=[('2 + 3','5'),('3 + 3','6'),('4 + 2','6'),('5 + 2','7'),('1 + 7','8'),('2 + 7','9'),('6 + 4','10'),('7 + 4','11')],demo=[('1 + 1','2'),('5 + 7','12')],question='What is {x}?',instruction='Calculate each sum.'),
 'subtraction':dict(query=[('8 - 3','5'),('9 - 3','6'),('7 - 2','5'),('6 - 2','4'),('10 - 1','9'),('10 - 2','8'),('9 - 5','4'),('8 - 5','3')],demo=[('3 - 2','1'),('12 - 1','11')],question='What is {x}?',instruction='Calculate each difference.'),
 'boolean_negation':dict(query=[('NOT true','false'),('NOT false','true'),('not(true)','false'),('not(false)','true'),('!true','false'),('!false','true'),('the opposite of true','false'),('the opposite of false','true')],demo=[('negate true','false'),('negate false','true')],question='What is the Boolean value of {x}?',instruction='Negate the indicated Boolean value. Return true or false.'),
 'month_successor':dict(query=[('January','February'),('February','March'),('March','April'),('April','May'),('May','June'),('June','July'),('July','August'),('August','September')],demo=[('October','November'),('November','December')],question='Which month comes immediately after {x}?',instruction='Give the month immediately following each input month.')}
 assert set(specs)==set(design['candidate_families']);assert not set(specs)&set(design['exclude_from_training']);pairs=[];overlap=[]
 for family,s in specs.items():
  qsrc={x for x,y in s['query']};dsrc={x for x,y in s['demo']};assert not qsrc&dsrc
  overlap.append(dict(family=family,exact_source_overlap=[],answer_overlap=sorted({y for x,y in s['query']}&{y for x,y in s['demo']}),note='Exact source items disjoint, not necessarily tokens or primitives. Arithmetic symbols/digits and Boolean values necessarily recur; answer-label reuse is not lexical independence.'))
  for fmt in design['formats']:
   ends=[]
   for source,answer in s['query']:
    prompt=('Question: '+s['question'].format(x=source)+'\nAnswer:') if fmt=='explicit_question_answer' else s['instruction']+'\n'+''.join(f'Input: {x}\nOutput: {y}\n' for x,y in s['demo'])+f'Input: {source}\nOutput:'
    ends.append(dict(source=source,answer=answer,prompt=prompt,ids=tok.encode(prompt)))
   for j in range(4):
    es=ends[j*2:j*2+2];valid=True;reasons=[]
    for e in es:
     groups=[]
     for target in es:
      group=[]
      for form in dict.fromkeys([target['answer'],target['answer'].capitalize(),target['answer'].upper()]):
       ids=tok.encode(e['prompt']+' '+form)
       if ids[:-1]==e['ids'] and len(ids)==len(e['ids'])+1:group.append(ids[-1])
      groups.append(sorted(set(group)))
     e['answer_groups']=groups
     if any(not g for g in groups):valid=False;reasons.append('No single-token answer group')
     if set(groups[0])&set(groups[1]):valid=False;reasons.append('Overlapping answer groups')
    pairs.append(dict(id=f'{family}:{fmt}:{j}',family=family,format=fmt,split='train',endpoints=es,valid=valid,invalid_reasons=sorted(set(reasons)),equal_length=len(es[0]['ids'])==len(es[1]['ids'])))
 assert len(pairs)==56;prompts=[e['prompt'] for p in pairs for e in p['endpoints']];assert len(set(prompts))==112
 result=dict(stage='frozen_before_native_forward',design_sha256=sha(V/'broader_training_screen_design.json'),source_sha256=sha(Path(__file__)),tokenizer='tiktoken gpt2',specs=specs,demonstration_overlap=overlap,pairs=pairs,declared_pairs=56,declared_endpoints=112,valid_pairs=sum(p['valid'] for p in pairs),scope='Training-only authored format test. Some sources reused from H1039; boolean syntax changed. Family-level historical comparison is not an exactly matched prompt comparison. Intended answer labels are conservative for open-ended sounds/colors.')
 op=V/'broader_training_screen_manifest.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print('FROZEN',result['valid_pairs'],'valid of56 pairs',sha(op))
if __name__=='__main__':main()
