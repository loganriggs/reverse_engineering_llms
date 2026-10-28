"""H882 predeclared multi-domain matched-source competence screen."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_multidomain_matched_competence'
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTHS=['January','February','March','April','May','June','July','August','September','October','November','December']
PAIRS=[]
def add(domain,sources,answers):PAIRS.append(dict(domain=domain,sources=sources,answers=answers))
for i in range(7):add('weekday_successor',[DAYS[i],DAYS[(i+3)%7]],[DAYS[(i+1)%7],DAYS[(i+4)%7]])
for i in range(12):add('month_successor',[MONTHS[i],MONTHS[(i+5)%12]],[MONTHS[(i+1)%12],MONTHS[(i+6)%12]])
for a,b in [('dog','dogs'),('cat','cats'),('book','books'),('tree','trees'),('report','reports'),('picture','pictures'),('bottle','bottles'),('key','keys')]:add('agreement',[a,b],['is','are'])
for a,b in [('hot','cold'),('big','small'),('good','bad'),('high','low'),('young','old'),('wet','dry'),('hard','soft'),('open','closed')]:add('antonym',[a,b],[b,a])
verbs=[('walk','walked'),('play','played'),('work','worked'),('look','looked'),('go','went'),('come','came'),('see','saw'),('take','took'),('give','gave'),('make','made'),('write','wrote'),('speak','spoke')]
for i in range(6):add('past_tense',[verbs[i][0],verbs[i+6][0]],[verbs[i][1],verbs[i+6][1]])
nouns=[('dog','dogs'),('cat','cats'),('book','books'),('tree','trees'),('car','cars'),('house','houses'),('man','men'),('woman','women'),('child','children'),('foot','feet'),('tooth','teeth'),('mouse','mice')]
for i in range(6):add('plural',[nouns[i][0],nouns[i+6][0]],[nouns[i][1],nouns[i+6][1]])
langs=[('Italy','Italian','Japan','Japanese'),('France','French','China','Chinese'),('Germany','German','Spain','Spanish'),('Russia','Russian','Greece','Greek'),('Poland','Polish','Turkey','Turkish'),('Finland','Finnish','Sweden','Swedish'),('Norway','Norwegian','Portugal','Portuguese'),('Denmark','Danish','Hungary','Hungarian')]
for x,a,y,b in langs:add('language',[x,y],[a,b])
for a,b in [('John','Mary'),('Alice','Bob'),('Sarah','Tom'),('David','Anna'),('Peter','Jane'),('James','Susan'),('Michael','Emma'),('Robert','Linda')]:add('binding',[a,b],[b,a])
DOMAINS=list(dict.fromkeys(p['domain'] for p in PAIRS));TARGETS=[(13,0),(14,4),(8,3)]
def render(domain,fmt,word,sources):
 a,b=sources
 templates={
 'weekday_successor':[f'The day after {word} is',f'The next day after {word} is'],
 'month_successor':[f'The month after {word} is',f'The next month after {word} is'],
 'agreement':[f'The {word} near the river',f'The {word} in the garden'],
 'antonym':[f'The opposite of {word} is',f'The antonym of {word} is'],
 'past_tense':[f'The past tense of {word} is',f'The simple past of {word} is'],
 'plural':[f'The plural of {word} is',f'The plural form of {word} is'],
 'language':[f'The main language spoken in {word} is',f'The official language of {word} is'],
 'binding':[f'Yesterday, {a} and {b} met. {word} sent a letter to',f'Earlier, {a} and {b} met. {word} gave a book to']}
 return templates[domain][0 if fmt=='A' else 1]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H882','hashes':hashes,'pairs':PAIRS,'targets':TARGETS,'design':'Eight predeclared domains, two templates each, each with matched repeated-token copying control. Source tokens same across tasks; only source changes across endpoints. No correctness-based pair exclusion. Eligibility jointly requires source/answers single-token and endpoint alignment in all four cells. New combinations/tasks relative to H873, some tokens/templates familiar from prior research; not pretraining-disjoint.','gate':'Per domain/family/format: >=4 eligible pairs, both-endpoint binary accuracy>=0.75, pooled full-vocabulary top1>=0.5. Allcells and failures retained; no tuning after outcomes.','next':'Only competent task/copy cell pairs eligible for mechanistic transfer; separately quantify single-example and vocabulary-wide effects. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples=[];excluded=[]
 for pair,p in enumerate(PAIRS):
  domain=p['domain'];sources=p['sources'];rows=[];errors=[];source_ids=[tok.encode(' '+v,add_special_tokens=False) for v in sources]
  if any(len(v)!=1 for v in source_ids):excluded.append({'pair':pair,'domain':domain,'reasons':['source_not_single']});continue
  for fmt in ['A','B']:
   for fam in ['copy','task']:
    answers=sources if fam=='copy' else p['answers'];rr=[]
    for order,word in enumerate(sources):
     text=(f'word {word}; word {word}; word' if fmt=='A' else f'item {word}, item {word}, item') if fam=='copy' else render(domain,fmt,word,sources)
     ids=tok.encode(text,add_special_tokens=False);answerids=[continuation_id(tok,text,' '+v) for v in answers]
     if any(v is None for v in answerids) or len(set(answerids))!=2:errors.append('answer_not_single_or_distinct');continue
     rr.append(dict(pair=pair,domain=domain,split='screen',format=fmt,query=0,family=fam,order=order,expected=order,text=text,ids=ids,groups=[[v] for v in answerids],answers=answers,id=f'{pair}-{fmt}-{fam}'))
    if len(rr)!=2:continue
    if len(rr[0]['ids'])!=len(rr[1]['ids']):errors.append('unequal_endpoint_length');continue
    changed=[i for i,(u,v) in enumerate(zip(rr[0]['ids'],rr[1]['ids'])) if u!=v]
    if not changed or any(rr[o]['ids'][i]!=source_ids[o][0] for i in changed for o in [0,1]):errors.append('non_source_or_absent_change');continue
    for r in rr:r['source_positions']=changed;r['source_words']=[tok.decode([r['ids'][i]]) for i in changed]
    rows.extend(rr)
  if errors or len(rows)!=8:excluded.append({'pair':pair,'domain':domain,'reasons':sorted(set(errors))});continue
  examples.extend(rows)
 assert examples;(ART/f'{S}_manifest.json').write_text(json.dumps({'examples':examples,'excluded':excluded},indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 stash={};hooks=[];originals={}
 for l,h in TARGETS:
  att=model.blocks[l].attn;original=att.squared_attention;originals[l]=original
  def capture(q,k,v,q2,k2,layer=l,head=h,fn=original):
   A=(torch.einsum('bhd,bthd->bht',q[:,-1],k)/128)*(torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/128);stash[layer,'routing']=A[:,head].cpu();stash[layer,'values']=v[:,:,head].cpu();return fn(q,k,v,q2,k2)
  att.squared_attention=capture
  def hook(mod,args,layer=l,head=h):stash[layer,'output']=args[0][:,-1,head*128:(head+1)*128].cpu()
  hooks.append(att.c_proj.register_forward_pre_hook(hook))
 buckets=defaultdict(list)
 for i,e in enumerate(examples):buckets[len(e['ids'])].append(i)
 native={};measurements={};captures={};buscaps={};actual=0;audit=0.;reconstruction=0.;static={};staticerr=0.
 for length,ii in sorted(buckets.items()):
  for st in range(0,len(ii),12):
   ix=ii[st:st+12];ids=torch.tensor([examples[i]['ids'] for i in ix],device='cuda');x0=torch.nn.functional.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
   for l,block in enumerate(model.blocks):
    x,bus=block(x,bus,x0)
    if l==0:captured_bus=bus.cpu()
   y=30*torch.tanh(model.lm_head(torch.nn.functional.rms_norm(x,(1152,))[:,-1]).float()/30);actual+=len(ix)
   if not native:
    snapshot={k:v.clone() for k,v in stash.items()};yy=final_logits(model,ids);actual+=len(ix);audit=float((yy-y).abs().max());stash=snapshot
   for k,i in enumerate(ix):
    native[i]=y[k].double().cpu();measurements[i]=measure(native[i],examples[i]['groups']);measurements[i]['predicted_token']=tok.decode([measurements[i]['top_token']]);buscaps[i]=captured_bus[k];captures[i]={}
    for l,h in TARGETS:
     A=stash[l,'routing'][k];V=stash[l,'values'][k];H=stash[l,'output'][k];pred=(A[:,None]*V).sum(0);reconstruction=max(reconstruction,float((pred-H).norm()/H.norm().clamp_min(1e-12)));captures[i][l,h]={'routing':A,'mixed_values':V,'head_output':H}
     for p in examples[i]['source_positions']:
      key=(int(ids[k,p]),h);val=captured_bus[k,p,h]
      if key in static:staticerr=max(staticerr,float((val-static[key]).abs().max()))
      else:static[key]=val.clone()
  print('DONE',length,flush=True)
 for hook in hooks:hook.remove()
 for l,fn in originals.items():model.blocks[l].attn.squared_attention=fn
 cells=[]
 for domain in DOMAINS:
  for fam in ['copy','task']:
   for fmt in ['A','B']:
    ix=[i for i,e in enumerate(examples) if (e['domain'],e['family'],e['format'])==(domain,fam,fmt)];by=defaultdict(list)
    for i in ix:by[examples[i]['pair']].append(i)
    both=[all((measurements[i]['margin']>0)==(examples[i]['expected']==0) for i in inds) for inds in by.values()];top=[measurements[i]['top1'][examples[i]['expected']] for i in ix];c=dict(domain=domain,family=fam,format=fmt,pairs=len(by),both_correct=float(np.mean(both)) if both else 0.,full_vocab=float(np.mean(top)) if top else 0.);c['pass']=c['pairs']>=4 and c['both_correct']>=.75 and c['full_vocab']>=.5;cells.append(c)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_forward_max_abs':audit,'signed_AV_relative_error':reconstruction,'source_bus_same_token_max_abs':staticerr,'eligible_lexical_pairs':len(examples)//8,'source_positions_aligned_within_task_pairs':True,'only_source_tokens_change_with_order':True,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(torch.isfinite(v).all().item() for v in native.values())};assert audits['hashes_unchanged'] and audit<1e-4 and reconstruction<1e-4 and staticerr<1e-4 and audits['finite'];torch.save({'examples':examples,'native':native,'captures':captures,'bus':buscaps,'static_bus':static},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'competence':cells,'excluded':excluded,'measurements':measurements,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'competence':cells,'audits':audits}),flush=True)
if __name__=='__main__':main()
