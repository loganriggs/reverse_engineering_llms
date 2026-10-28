"""H962 fixed 16-family competence screen for expanded causal supervision."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import groups,measure,forward
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from transformers import AutoTokenizer
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_expanded_task_bank'
def suite():
 names=[('John','Mary'),('Alice','Bob'),('Carol','Dave'),('Sarah','Tom'),('Peter','Jane'),('David','Kate'),('Paul','Lisa'),('Henry','Rose')]
 nouns=[('dog','dogs'),('cat','cats'),('bird','birds'),('horse','horses'),('pilot','pilots'),('chef','chefs'),('farmer','farmers'),('singer','singers')]
 adjs=[('wonderful','terrible'),('excellent','awful'),('great','bad'),('pleasant','unpleasant'),('brilliant','boring'),('enjoyable','tedious'),('lovely','ugly'),('satisfying','disappointing')]
 colors=['red','blue','green','yellow','black','white','orange','purple'];countries=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo'),('Spain','Madrid'),('China','Beijing'),('Russia','Moscow'),('England','London')];rows=[]
 for fmt in ['A','B']:
  for i,((a,b),(sing,pl),(pos,neg)) in enumerate(zip(names,nouns,adjs)):
   def add(f,p,ans,change):
    assert len(p)==2 and len(ans)==2 and ans[0]!=ans[1];rows.append(dict(id=f'{f}-{fmt}-{i}',family=f,format=fmt,source=i,split=('discovery' if i<4 else 'lexical') if fmt=='A' else ('structural' if i<4 else 'joint'),whole_task_holdout=f in ['sequence_copy','role_binding','spatial_inverse','tense'],prompts=p,answers=ans,intervention_variable=change))
   add('agreement',[f'The {n} '+('near the window' if fmt=='A' else 'beside the door') for n in [sing,pl]],['is','are'],'subject number')
   add('sentiment',[f'Review: The '+('film' if fmt=='A' else 'performance')+f' was {v}. Sentiment:' for v in [pos,neg]],['positive','negative'],'evaluative adjective polarity')
   add('negation',[f'Review: The '+('film' if fmt=='A' else 'performance')+f' was {prefix}{pos}. Sentiment:' for prefix in ['', 'not ']],['positive','negative'],'presence of negation; length changes allowed for competence')
   add('entity_binding',[f'{a} and {b} went to the '+('store' if fmt=='A' else 'park')+f'. {giver} '+('gave the book to' if fmt=='A' else 'handed the bag to') for giver in [a,b]],[b,a],'giver identity with recipient as completion')
   x,y=colors[i],colors[(i+3)%8]
   add('attribute_binding',[f'{a} has a {x} hat. {b} has a {y} hat. The color of {who}\'s hat is' if fmt=='A' else f'{a}\'s bag is {x}; {b}\'s bag is {y}. {who}\'s bag is' for who in [a,b]],[x,y],'queried entity; fixed attribute assignments')
   add('role_binding',[f'{agent} chased {patient}. The person who did the chasing was' if fmt=='A' else f'{patient} was followed by {agent}. The person doing the following was' for agent,patient in [(a,b),(b,a)]],[a,b],'agent/patient swap, active/passive templates')
   add('spatial_inverse',[f'{a} is to the left of {b}. {b} is to the' if direction==0 else f'{a} is to the right of {b}. {b} is to the' for direction in [0,1]] if fmt=='A' else [f'The {x} box is {rel} the {y} box. The {y} box is' for rel in ['above','below']],['right','left'] if fmt=='A' else ['below','above'],'inverse spatial relation with varied entity contexts')
   add('temporal_order',[f'{first} arrived before {second}. The first to arrive was' if fmt=='A' else f'{second} left after {first}. The first to leave was' for first,second in [(a,b),(b,a)]],[a,b],'event ordering')
   lo=1+i;hi=lo+1
   add('numerical_comparison',[f'The {q} of {lo} and {hi} is' for q in (['larger','smaller'] if fmt=='A' else ['greater','lesser'])],[str(hi),str(lo)],'comparison operation with fixed numbers')
   v=i%4+1;w=(i//4)+1
   add('arithmetic',[f'{v} {op} {w} =' if fmt=='A' else f'What is {v} {op} {w}? Answer:' for op in ['+','-']],[str(v+w),str(v-w)],'addition versus subtraction')
   add('key_value',[f'{a}: {x}; {b}: {y}; {who}:' if fmt=='A' else f'{a} = {x}, {b} = {y}, {who} =' for who in [a,b]],[x,y],'queried key with fixed values')
   key=colors[(i+5)%8];sep=';' if fmt=='A' else ','
   add('sequence_copy',[f'{key} {val}{sep} {key} {val}{sep} {key}' for val in [x,y]],[x,y],'repeated sequence value')
   add('pronoun_gender',[f'The {person} near the {sing} said that' if fmt=='A' else f'The {person} beside the {sing} explained that' for person in ['man','woman']],['he','she'],'referent gender with varied noun contexts')
   add('tense',[f'{when}, {a}' if fmt=='A' else f'{a} said that {when.lower()} he' for when in ['Yesterday','Tomorrow']],['was','will'],'temporal cue; broad completion not unique tense marker')
   articlepairs=[('apple','banana'),('orange','pear'),('egg','cake'),('umbrella','coat'),('elephant','horse'),('island','village'),('ocean','river'),('engine','motor')]
   va,vb=articlepairs[i];add('article_sound',[f'The word is "{word}". With an indefinite article:' if fmt=='A' else f'Complete with a or an before "{word}":' for word in [va,vb]],['an','a'],'vowel versus consonant onset')
   c1,cap1=countries[i];c2,cap2=countries[(i+1)%8];add('capital_recall',[f'The capital of {country} is' if fmt=='A' else f'{country} has its capital in' for country in [c1,c2]],[cap1,cap2],'country identity')
 return rows
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'tokenizer_helper':ROOT/'scripts/run_bilin18_broad_task_competence.py','forward_helper':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};rows=suite();assert len({tuple(r['prompts']) for r in rows})==len(rows), 'duplicate prompt pair'
 cfg={'experiment':'H962','hashes':hashes,'design':'Fixed16families x8 source slots x2formats, paired answer-changing examples. Native competence only; no circuit/featurizer claims. Save all outcomes and tokenization exclusions. No equal-length filter at competence stage; alignment handled later.','split':'A first4 source discovery, A last4 lexical, B first4 structural, B last4 joint. Whole-task training holdouts predeclared: sequence_copy,role_binding,spatial_inverse,tense. These are development holdouts, not the original reserved test.','gate':'Per family/split >=4 DISTINCT prompt pairs, >=.75 both-endpoint contrast correctness, >=.5 intended group full-vocabulary top1. Duplicated templates counted once; all are reported, no posthoc prompt replacement.','groups':'Existing fixed agreement/sentiment/negation semantic groups; otherwise exact answer tokens. Strict single-token continuation boundary check; excluded pairs retained in report.','limits':'Some prompts reuse prior research. Eight source slots do not imply lexical independence across all task families. Semantic ambiguities and tokenization can make a format unsuitable without proving inability.','purpose':'Screen breadth cheaply before expensive causal targets; no new training.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');mh=sha(ART/f'{S}_manifest.json');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex=[];pairs=[];excluded=[];seen={}
 for row in rows:
  texts=row['prompts'];groupstrings=groups(row['family'],row['answers']);ee=[]
  for variant,t in enumerate(texts):
   ids=tok.encode(t,add_special_tokens=False);gg=[[continuation_id(tok,t,s) for s in g] for g in groupstrings];gg=[[x for x in g if x is not None] for g in gg]
   if any(not g for g in gg) or set(gg[0])&set(gg[1]):break
   ee.append(dict(id=row['id'],variant=variant,text=t,ids=ids,groups=gg))
  if len(ee)!=2:excluded.append(dict(id=row['id'],reason='single-token answer boundary or overlapping groups'));continue
  key=tuple(texts);duplicate=key in seen;prior=seen.get(key);seen[key]=row['id'];ii=list(range(len(ex),len(ex)+2));pairs.append(row|dict(indices=ii,duplicate=duplicate,duplicate_of=prior,equal_length=len(ee[0]['ids'])==len(ee[1]['ids']),same_final_token=ee[0]['ids'][-1]==ee[1]['ids'][-1]));ex.extend(ee)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};actual=0
 for length in sorted({len(e['ids']) for e in ex}):
  ii=[i for i,e in enumerate(ex) if len(e['ids'])==length]
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];ids=torch.tensor([ex[i]['ids'] for i in inds],device='cuda');y,_=forward(model,ids);actual+=len(inds)
   for k,i in enumerate(inds):native[i]=y[k].double().cpu()
 records=[]
 for row in pairs:
  measures=[measure(native[i],ex[i]['groups']) for i in row['indices']];records.append(row|dict(both_correct=measures[0]['margin']>0 and measures[1]['margin']<0,full_vocab=sum(measures[k]['top1'][k] for k in [0,1])/2,endpoint_metrics=measures))
 summary=[]
 for family in sorted({r['family'] for r in rows}):
  for split in ['discovery','lexical','structural','joint']:
   rr=[r for r in records if r['family']==family and r['split']==split];distinct={tuple(r['prompts']):r for r in rr};dd=list(distinct.values());both=float(np.mean([r['both_correct'] for r in dd])) if dd else 0;full=float(np.mean([r['full_vocab'] for r in dd])) if dd else 0
   summary.append(dict(family=family,split=split,declared_pairs=4,valid_pairs=len(rr),distinct_pairs=len(dd),both_correct=both,full_vocab=full,aligned_pairs=sum(r['equal_length'] and r['same_final_token'] for r in dd),pass_gate=len(dd)>=4 and both>=.75 and full>=.5,whole_task_holdout=next(r['whole_task_holdout'] for r in rows if r['family']==family)))
 # Independent repeat of the first token-valid pair checks deterministic replay.
 check=ex[0];y,_=forward(model,torch.tensor([check['ids']],device='cuda'));actual+=1;replay=float((y[0].double().cpu()-native[0]).abs().max())
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),manifest_unchanged=sha(ART/f'{S}_manifest.json')==mh,whole_model_forwards=actual,expected_forwards=len(ex)+1,replay_logits_max_abs=replay,finite=all(np.isfinite(m['margin']) for r in records for m in r['endpoint_metrics']),test_evaluations=0,no_training=True)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and audits['finite'] and actual==audits['expected_forwards'] and replay<1e-4
 torch.save(dict(examples=ex,native=native),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,summary=summary,records=records,excluded=excluded,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(summary,excluded,audits,flush=True)
if __name__=='__main__':main()
