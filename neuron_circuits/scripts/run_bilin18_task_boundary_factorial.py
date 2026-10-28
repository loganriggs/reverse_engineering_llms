"""H582: locked task-format factorial with lexical transfer and unfiltered failures."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_writer_transfer import suite as old_suite
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,measure,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_task_boundary_factorial'
def suite():
 names0=[('Andrew','Rachel'),('Patrick','Rebecca'),('Stephen','Diana'),('Anthony','Julia'),('Christopher','Victoria'),('Jonathan','Elizabeth'),('William','Michelle'),('Charles','Jennifer')]
 names1=[('Edward','Christine'),('Donald','Margaret'),('Kenneth','Patricia'),('Ronald','Sharon'),('Raymond','Catherine'),('Dennis','Jacqueline'),('Joseph','Angela'),('Douglas','Barbara')]
 nouns0=[('actor','actors'),('pilot','pilots'),('nurse','nurses'),('lawyer','lawyers'),('chef','chefs'),('guard','guards'),('dancer','dancers'),('painter','painters')]
 nouns1=[('sailor','sailors'),('baker','bakers'),('mechanic','mechanics'),('clerk','clerks'),('scientist','scientists'),('musician','musicians'),('waiter','waiters'),('engineer','engineers')]
 countries0=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo'),('Spain','Madrid'),('China','Beijing'),('Russia','Moscow'),('England','London')]
 countries1=[('Austria','Vienna'),('Greece','Athens'),('Ireland','Dublin'),('Portugal','Lisbon'),('Norway','Oslo'),('Sweden','Stockholm'),('Finland','Helsinki'),('Denmark','Copenhagen')]
 adjs0=[('wonderful','terrible'),('excellent','awful'),('delightful','dreadful'),('enjoyable','boring'),('great','bad'),('amazing','horrible'),('superb','poor'),('fantastic','disappointing')]
 adjs1=[('pleasant','unpleasant'),('brilliant','mediocre'),('entertaining','tedious'),('impressive','abysmal'),('lovely','ugly'),('satisfying','frustrating'),('outstanding','pathetic'),('perfect','lousy')]
 rows=[]
 for split,names,nouns,countries,adjs in [('discovery',names0,nouns0,countries0,adjs0),('transfer',names1,nouns1,countries1,adjs1)]:
  for i,(a,b) in enumerate(names):
   for f0 in (0,1):
    for f1 in (0,1):
     condition=f'{f0}{f1}'
     def add(family,prompts,answers,nuisance):rows.append(dict(id=f'{family}-{split}-{condition}-{i}',family=family,split=split,condition=condition,item=i,prompts=prompts+nuisance,answers=answers))
     prep=['beside','near'][f0];tail=['',' usually'][f1];ap=lambda n,o:f'The {n} {prep} the {o}{tail}'
     add('agreement',[ap(n,'window') for n in nouns[i]],['is','are'],[ap(n,'door') for n in nouns[i]])
     prefix=['national capital','capital city'][f0];tail=[' is',' is the city of'][f1];fp=lambda c:f'The {prefix} of {c}{tail}';cs=[countries[i],countries[(i+1)%8]]
     add('capital_recall',[fp(c) for c,ca in cs],[ca for c,ca in cs],['Fact: '+fp(c) for c,ca in cs])
     place=['school','library'][f0];verb=['gave','handed'][f1];bp=lambda g,obj:f'{a} and {b} visited the {place}. {g} {verb} the {obj} to'
     add('entity_binding',[bp(x,'notebook') for x in (a,b)],[b,a],[bp(x,'folder') for x in (a,b)])
     prefix=['Text','Review'][f0];label=['Overall sentiment','Sentiment'][f1];sp=lambda a,o:f'{prefix}: The {o} was {a}. {label}:'
     add('sentiment',[sp(a,'performance') for a in adjs[i]],['positive','negative'],[sp(a,'show') for a in adjs[i]])
     subj=['He','She'][f0];verb=['spoke with','talked to'][f1];cp=lambda x,o:f'{subj} {verb} {x} about the {o}. {subj} {verb} {x} about the {o}. {subj} {verb}'
     add('sequence_copy',[cp(x,'letter') for x in (a,b)],[a,b],[cp(x,'message') for x in (a,b)])
 return rows
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_diverse_writer_transfer.json';cfg=dict(experiment='H582',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),design='Five task-specific 2x2 format factorials, eight discovery pairs and eight lexical-transfer pairs; both answer endpoints plus answer-preserving nuisance versions. Original H581 failures remain recorded. Some factorial cells repeat H581 prompts; new combinations are not claimed globally novel.',factors={'agreement':['beside/near','empty/usually suffix'],'capital_recall':['national capital/capital city','is/is the city of suffix'],'entity_binding':['school/library','gave/handed'],'sentiment':['Text/Review','Overall sentiment/Sentiment'],'sequence_copy':['He/She','spoke with/talked to']},gate='>=4 pairs; both-endpoint candidate accuracy >=.75; intended answer-group full-vocabulary top1 >=.5. Report nuisance separately.',selection='Choose lexicographically first passing discovery condition from 00,01,10,11, before inspecting transfer scores. If none passes, none selected. All cells retained.',scope='Changes can alter computation as well as answer boundary. Factorial effects are conditional on each task and do not isolate a universal formatting variable. Candidate groups fixed from H533; no post-result expansion.',hypotheses=['Recall boundary completion should improve direct answer token generation.','Sentiment cue words interact rather than acting as universal format decoration.','Agreement and binding failures need not be solved by small wording changes.','A discovery-passing condition may fail lexical transfer.'])
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));rows=suite();mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,x) for x in g] for g in groups(r['family'],r['answers'])];gs=[[x for x in g if x is not None] for g in gs]
   if any(not g for g in gs) or set(gs[0])&set(gs[1]):break
   ee.append(dict(**r,variant=v,text=text,ids=tok.encode(text,add_special_tokens=False),groups=gs))
  if len(ee)!=4:excluded.append(r['id']);continue
  for e in ee:examples[e['id'],e['variant']]=e
 print('LOCKED',len(examples),'prompts',flush=True);model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 result=dict(config=cfg,manifest_sha256=mh,excluded=excluded,native=[],competence={},selected={},status='running');native={};savedlogits={}
 for split in ('discovery','transfer'):
  buckets=defaultdict(list)
  for k,e in examples.items():
   if e['split']==split:buckets[len(e['ids'])].append(k)
  for T,kk in buckets.items():
   for st in range(0,len(kk),24):
    bb=kk[st:st+24];zz,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
    for ix,k in enumerate(bb):native[k]=measure(zz[ix],examples[k]['groups']);savedlogits[k]=zz[ix].cpu();result['native'].append(dict(id=k[0],variant=k[1],top_text=tok.decode([native[k]['top_token']]),**native[k]))
  for fam in sorted({r['family'] for r in rows}):
   for cond in ('00','01','10','11'):
    rr=[r for r in rows if r['split']==split and r['family']==fam and r['condition']==cond and (r['id'],0) in native];n=len(rr);both=float(np.mean([native[r['id'],0]['margin']>0 and native[r['id'],1]['margin']<0 for r in rr]));top=float(np.mean([native[r['id'],v]['top1'][v] for r in rr for v in (0,1)]));nuis=float(np.mean([native[r['id'],v+2]['top1'][v] for r in rr for v in (0,1)]));result['competence'][f'{fam}_{split}_{cond}']=dict(pairs=n,both_endpoint_choice=both,full_vocab_top1=top,nuisance_full_vocab_top1=nuis,passes=bool(n>=4 and both>=.75 and top>=.5))
   if split=='discovery':result['selected'][fam]=next((c for c in ('00','01','10','11') if result['competence'][f'{fam}_discovery_{c}']['passes']),None)
  if split=='discovery':
   (ART/f'{STEM}_locked_selection.json').write_text(json.dumps(result['selected'],indent=2));print('SELECTED BEFORE TRANSFER',result['selected'],flush=True)
 old=json.loads(src.read_text());oldrows=json.loads((ART/'bilin18_diverse_writer_transfer_manifest.json').read_text());oldnative={(n['id'],n['variant']):n for n in old['native']};oldtext={(text,tuple(r['answers'])):oldnative[r['id'],v] for r in oldrows for v,text in enumerate(r['prompts'])};replays=[abs(native[k]['margin']-oldtext[e['text'],tuple(e['answers'])]['margin']) for k,e in examples.items() if (e['text'],tuple(e['answers'])) in oldtext];result['audits']=dict(repeated_prompt_count=len(replays),repeat_margin_max_error=max(replays),finite=all(np.isfinite(n['margin']) for n in native.values()),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),selection_matches_locked=result['selected']==json.loads((ART/f'{STEM}_locked_selection.json').read_text()));result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(examples=examples,logits=savedlogits),ART/f'{STEM}_cache.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
