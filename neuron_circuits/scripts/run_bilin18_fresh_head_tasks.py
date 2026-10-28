"""H732: locked fresh wording/lexicon panel with native and fixed-head controls."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_fresh_head_tasks'
def suite():
 rows=[]
 names=[('Ben','Nora'),('Sam','Julia'),('Max','Clara'),('Leo','Diana'),('Luke','Mia'),('Adam','Sophie'),('Eric','Rachel'),('Nathan','Olivia'),('Alex','Chloe'),('Oscar','Eva'),('Victor','Irene'),('Simon','Zoe'),('Felix','Amy'),('Hugo','Linda'),('Owen','Ruth'),('Ethan','Megan')]
 nouns=['chair','window','bridge','lamp','clock','table','door','basket','car','truck','boat','train','desk','shelf','box','shoe'];plurals=['chairs','windows','bridges','lamps','clocks','tables','doors','baskets','cars','trucks','boats','trains','desks','shelves','boxes','shoes']
 words=['apple','pear','peach','plum','lemon','lime','grape','melon','horse','goat','sheep','wolf','bear','fox','lion','tiger']
 countries=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo'),('Spain','Madrid'),('China','Beijing'),('Russia','Moscow'),('England','London'),('Austria','Vienna'),('Greece','Athens'),('Ireland','Dublin'),('Portugal','Lisbon'),('Norway','Oslo'),('Sweden','Stockholm'),('Finland','Helsinki'),('Denmark','Copenhagen')]
 adjectives=[('helpful','useless'),('reliable','unreliable'),('friendly','rude'),('efficient','inefficient'),('beautiful','hideous'),('comfortable','uncomfortable'),('exceptional','atrocious'),('magnificent','miserable'),('charming','repulsive'),('splendid','appalling'),('remarkable','inferior'),('marvelous','horrendous'),('terrific','horrific'),('fabulous','wretched'),('exquisite','disgusting'),('stellar','deplorable')]
 def add(f,i,p,a,n):rows.append(dict(id=f'h732-{f}-{i:02}',family=f,split='format_a' if i<8 else 'format_b',prompts=p,answers=a,nuisance=n))
 for i,((a,b),noun,plural,(pos,neg)) in enumerate(zip(names,nouns,plurals,adjectives)):
  other=nouns[(i+3)%16];prep='behind' if i<8 else 'in front of';add('agreement',i,[f'The {n} {prep} the {other}' for n in [noun,plural]],['is','are'],f'The {noun} {prep} the {nouns[(i+5)%16]}')
  place='museum' if i<8 else 'station';obj='ticket' if i<8 else 'parcel';add('entity_binding',i,[f'{a} and {b} visited the {place}. {giver} passed the {obj} to' for giver in [a,b]],[b,a],f'{a} and {b} visited the {place}. {a} passed the envelope to')
  key=words[i];v,w=words[(i+2)%16],words[(i+5)%16];sep=' / ' if i<8 else ' | ';add('sequence_copy',i,[sep.join([f'{key} {x}',f'{key} {x}',key]) for x in [v,w]],[v,w],sep.join([f'{words[(i+7)%16]} {v}',f'{words[(i+7)%16]} {v}',words[(i+7)%16]]))
  c,cap=countries[i];c2,cap2=countries[(i+5)%8+(0 if i<8 else 8)];template='Country: {}\nCapital:' if i<8 else 'Question: Which city is the capital of {}?\nAnswer:';add('capital_recall',i,[template.format(c),template.format(c2)],[cap,cap2],'Geography\n'+template.format(c))
  thing='service' if i<8 else 'product';template='Customer feedback: The '+thing+' was {}.\nSentiment:';add('sentiment',i,[template.format(pos),template.format(neg)],['positive','negative'],template.format(pos).replace('Customer feedback:','Customer review:'))
 return rows
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();rows=suite();manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));paths={'manifest':manifest,'old_manifest':ART/'bilin18_group_task_transfer_manifest.json','model':SNAPSHOT/'pytorch_model.bin','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py','answer_groups':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py'};hashes={k:sha(p) for k,p in paths.items()};oldrows=json.loads(paths['old_manifest'].read_text());oldtexts={t for r in oldrows for t in r['prompts']+[r['nuisance']]};assert not any(t in oldtexts for r in rows for t in r['prompts']+[r['nuisance']]);sites={'agreement_head':(11,3),'capital_head':(13,0),'binding_reference':(14,4),'copy_reference':(8,3)}
 cfg=dict(experiment='H732',hashes=hashes,sites=sites,design='Lock80pairs,5families,2formatblocks of8,3variants before modeloutputs. Newagreementnouns,bindingnames,copyvocabulary,sentimentadjectives; factualpairs reuse establishedfacts withnewwording/pairing, notfact-heldout. Nativecompetence andall4previouslyfixedheads across allfamilies, no headselection orindividualfiltering. Bothcounterfactual directions,nuisance0<-2,identity0<-0. Captureallnativeheadcoordinates for subsequentfrozenfeaturizertests.',competence_gate='Perfamily/format: >=4tokenvalidpairs, >=.75both-endpoint contrastcorrect, >=.5pooled fullvocab intendedsettop1. Report nuisancechoice separately. No filtering on effects.',limitations='Hand-authored finitepanel; formatblocks areevaluation strata, notnewdiscovery/selectionsplit. Wordingfresh versusH717manifest only, notcertifiedtrainingunseen. Capitalfacts intentionallyoverlap. Headswaps architectural, notsemanticunits. Zero reservednaturaltest use.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};excluded=[]
 for row in rows:
  enc=[]
  for v,text in enumerate(row['prompts']+[row['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(row['family'],row['answers'])];gs=[[a for a in g if a is not None] for g in gs]
   if not all(gs) or set(gs[0])&set(gs[1]):break
   enc.append(dict(ids=tok.encode(text,add_special_tokens=False),groups=gs,family=row['family'],split=row['split']))
  if len(enc)!=3:excluded.append(row['id']);continue
  for v,e in enumerate(enc):examples[row['id'],v]=e
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};heads={};buckets=defaultdict(list);forwards=0
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];cap={};hooks=[]
   for site,(layer,head) in sites.items():
    def pre(m,args,site=site,head=head):cap[site]=args[0][:,-1,head*128:(head+1)*128].cpu()
    hooks.append(model.blocks[layer].attn.c_proj.register_forward_pre_hook(pre))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb)
   for n,k in enumerate(bb):native[k]=logits[n].cpu();heads[k]={s:c[n] for s,c in cap.items()}
 native_rows=[];summary={}
 for k,e in examples.items():native_rows.append(dict(id=k[0],variant=k[1],family=e['family'],split=e['split'],**measure(native[k],e['groups'])))
 for family in sorted({r['family'] for r in rows}):
  summary[family]={}
  for split in ['format_a','format_b']:
   rr=[r for r in rows if r['family']==family and r['split']==split and (r['id'],0) in examples];both=[];top=[];nuis=[]
   for r in rr:
    ms=[measure(native[r['id'],v],examples[r['id'],v]['groups']) for v in range(3)];both.append(ms[0]['margin']>0 and ms[1]['margin']<0);top.extend([ms[v]['top1'][v] for v in [0,1]]);nuis.append(ms[2]['margin']>0)
   s=dict(pairs=len(rr),both_endpoint_choice=float(np.mean(both)) if rr else 0,full_vocab_top1=float(np.mean(top)) if rr else 0,nuisance_choice=float(np.mean(nuis)) if rr else 0);s['pass']=s['pairs']>=4 and s['both_endpoint_choice']>=.75 and s['full_vocab_top1']>=.5;summary[family][split]=s
 print('NATIVE',summary,'EXCLUDED',excluded,flush=True);records=[]
 for site,(layer,head) in sites.items():
  jobs=defaultdict(list)
  for row in rows:
   if (row['id'],0) not in examples:continue
   for rv,dv in [(0,1),(1,0),(0,2),(0,0)]:jobs[len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,site=site))
  for length,jj in jobs.items():
   for lo in range(0,len(jj),12):
    bb=jj[lo:lo+12];replacement=torch.stack([heads[r['id'],r['donor']][site] for r in bb]).cuda()
    def edit(m,args):
     y=args[0].clone();y[:,-1,head*128:(head+1)*128]=replacement.to(y);return (y,)
    hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
    try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
    finally:hook.remove()
    forwards+=len(bb)
    for n,r in enumerate(bb):
     k=(r['id'],r['receiver']);gs=examples[k]['groups'];base=measure(native[k],gs);ed=measure(logits[n],gs);change=ed['margin']-base['margin'];sign=1 if r['donor'] in [0,2] else -1;records.append(dict(**r,margin_change=change,donor_directed_margin_change=sign*change,edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']]))
  print('DONE',site,flush=True)
 identity=max(abs(r['margin_change']) for r in records if r['receiver']==r['donor']);audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['margin_change']) for r in records),identity_max_margin_error=identity,whole_model_forwards=forwards,native_examples=len(native),no_exact_prompt_overlap_with_h717=True,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and identity<1e-4;torch.save(dict(native_logits=native,head_coordinates=heads,examples=examples),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,excluded_pairs=excluded,native=native_rows,summary=summary,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
