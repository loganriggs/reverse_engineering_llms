#!/usr/bin/env python3
"""H535: frozen H534 write transport across output/readout formats."""
import json,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,continuation_id,measure
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_task_readout_crossing'
torch.set_num_threads(6)

def suite():
 rows=[]
 nouns=[('engineer','engineers'),('scientist','scientists'),('musician','musicians'),('manager','managers'),('lawyer','lawyers'),('chef','chefs'),('painter','painters'),('dancer','dancers')]
 adjectives=[('excellent','terrible'),('wonderful','awful'),('great','bad'),('delightful','dreadful'),('amazing','horrible'),('superb','poor'),('fantastic','disappointing'),('enjoyable','boring')]
 def add(fam,readout,i,texts,answers,expected=(0,1),kind='task'):
  rows.append(dict(id=f'{fam}-{readout}-{i}',family=fam,readout=readout,item=i,prompts=texts,groups=answers,expected=list(expected),kind=kind))
 for i,(sg,pl) in enumerate(nouns):
  stems=[f'The {n} near the door' for n in (sg,pl)]
  add('agreement','original_verbs',i,stems,groups('agreement',['is','are']))
  add('agreement','works_work',i,[s+' usually' for s in stems],[[' works'],[' work']])
  add('agreement','seems_seem',i,[s+' often' for s in stems],[[' seems'],[' seem']])
  add('agreement','pronouns',i,[s+' needed help because' for s in stems],[[' he',' she'],[' they']])
  add('agreement','reflexives',i,[s+' decided to introduce' for s in stems],[[' himself',' herself'],[' themselves']])
  add('agreement','modal_invariance',i,[s+' can' for s in stems],[[' works'],[' work']],(1,1),'invariance')
 for i,(pos,neg) in enumerate(adjectives):
  stems=[f'Review: The concert was {w}.' for w in (pos,neg)]
  add('sentiment','original_labels',i,[s+' Sentiment:' for s in stems],groups('sentiment',['positive','negative']))
  add('sentiment','enjoyment',i,[s+' I' for s in stems],[[' enjoyed',' loved'],[' hated',' disliked']])
  add('sentiment','recommendation',i,[s+' I would' for s in stems],[[' recommend'],[' avoid']])
  add('sentiment','rating',i,[s+' Rating:' for s in stems],[[' 5',' five'],[' 1',' one']])
  add('sentiment','recommend_yes_no',i,[s+' Would you recommend it? Answer:' for s in stems],[[' Yes',' yes'],[' No',' no']])
  add('sentiment','avoid_yes_no',i,[s+' Would you avoid it? Answer:' for s in stems],[[' Yes',' yes'],[' No',' no']],(1,0))
  add('sentiment','opposite_labels',i,[s+' The opposite sentiment is' for s in stems],groups('sentiment',['positive','negative']),(1,0))
 return rows

@torch.inference_mode()
def run(model,ids,vector=None,site='mlp'):
 d=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None
 for l,block in enumerate(model.blocks):
  h=None
  if l==12 and vector is not None and site=='mlp':
   def hook(m,args,out):
    out=out.clone();out[:,-1]+=vector.to(out);return out
   h=block.mlp.register_forward_hook(hook)
  try:x,bus=block(x,bus,x0)
  finally:
   if h:h.remove()
 if vector is not None and site=='final':x=x.clone();x[:,-1]+=vector.to(x)
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30)

@torch.inference_mode()
def main():
 start=time.time();rows=suite();source=ART/'bilin18_task_operand_transport_frozen_vectors.pt'
 cfg={'experiment':'H535','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'layer':12,'gate':'Per readout: eight declared pairs; at least 75% both-endpoint contrast correctness and >=50% pooled full-vocab top1. Invariance prompts have same correct class and are not counterfactual competence tests.','selection':'All formats and examples reported; no refitting H534 shared vectors.','control_modes':['shared','shared_reversed','nuisance_norm_matched','output_bias','shared_readout_removed','identity']+[f'random_{i:02}' for i in range(4)],'sites':'All modes at MLP12; shared and output_bias also at final residual.','output_bias':'Mean unembedding rows for original class1 minus class0, normalized to frozen shared norm.','shared_readout_removed':'Project shared vector off span of all original answer-token unembedding rows, then rescale to original norm. This only removes direct linear final-readout overlap, not nonlinear downstream effects.','precision':'float32 computation, shared loader bfloat16-rounded weights','hypotheses':['Frozen task writes transport to competent formats with disjoint output tokens.','Semantic orientation follows inverted questions using identical yes/no tokens.','Task writes exceed simple output-readout controls.'],'scope':'Fresh prompts but familiar lexical items and deliberately constructed templates, not random population or model-training holdout.'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(manifest.read_bytes()).hexdigest();print('LOCKED',len(rows),'pairs',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda')
 for module in (model.embedding,model.blocks,model.lm_head):
  module.float().eval()
  for p in module.parameters():p.requires_grad_(False)
 old=torch.load(source,weights_only=True,map_location='cpu');vectors={};geometry={}
 for fam in ('agreement','sentiment'):
  vv={n:old[fam][n] for n in cfg['control_modes'] if n in old[fam]};v=vv['shared'];gs=[[tok.encode(t,add_special_tokens=False)[0] for t in g] for g in groups(fam,['is','are'] if fam=='agreement' else ['positive','negative'])]
  w=model.lm_head.weight.detach().cpu().double();bias=w[gs[1]].mean(0)-w[gs[0]].mean(0);bias=(bias/bias.norm()*v.norm()).float()
  mat=w[gs[0]+gs[1]];_,sing,vh=torch.linalg.svd(mat,full_matrices=False);q=vh[sing>sing.max()*1e-10];removed=v.double()-q.T@(q@v.double());fraction=float(removed.norm()/v.norm());projected=(removed/removed.norm()*v.norm()).float()
  vv.update(output_bias=bias,shared_readout_removed=projected,identity=torch.zeros_like(v));vectors[fam]=vv;geometry[fam]={'retained_norm_fraction':fraction,'removed_span_rank':len(q),'original_output_ids':gs,'shared_bias_cosine':float(F.cosine_similarity(v,bias,dim=0))}
 torch.save(vectors,ART/f'{STEM}_frozen_vectors.pt')
 examples=[];excluded=[]
 for r in rows:
  for variant,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,t) for t in g] for g in r['groups']];gs=[[t for t in g if t is not None] for g in gs]
   if any(not g for g in gs):excluded.append((r['id'],variant));continue
   examples.append(dict(row=r,variant=variant,ids=tok.encode(text,add_special_tokens=False),groups=gs))
 native={};logps={};buckets=defaultdict(list)
 for e in examples:buckets[len(e['ids'])].append(e)
 for length,ee in sorted(buckets.items()):
  for begin in range(0,len(ee),12):
   batch=ee[begin:begin+12];logits=run(model,torch.tensor([e['ids'] for e in batch],device='cuda'))
   for ix,e in enumerate(batch):
    key=(e['row']['id'],e['variant']);mm=measure(logits[ix],e['groups']);mm['predicted_text']=tok.decode([mm['top_token']]);native[key]=mm;logps[key]=logits[ix].log_softmax(0).cpu()
 summary={}
 for fam in ('agreement','sentiment'):
  summary[fam]={}
  for readout in sorted(set(r['readout'] for r in rows if r['family']==fam)):
   rr=[r for r in rows if r['family']==fam and r['readout']==readout and all((r['id'],v) in native for v in (0,1))]
   both=[all((native[(r['id'],v)]['margin']>0)==(r['expected'][v]==0) for v in (0,1)) for r in rr];top=[native[(r['id'],v)]['top1'][r['expected'][v]] for r in rr for v in (0,1)]
   ss={'pairs':len(rr),'both_endpoint_choice':float(np.mean(both)),'full_vocab_top1':float(np.mean(top)),'kind':rr[0]['kind'] if rr else None};ss['pass']=len(rr)>=4 and ss['both_endpoint_choice']>=.75 and ss['full_vocab_top1']>=.5;summary[fam][readout]=ss
 result={'config':cfg,'manifest_sha256':mh,'summary':summary,'geometry':geometry,'excluded':excluded,'native':[{'id':k[0],'variant':k[1],**v} for k,v in native.items()],'interventions':[],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 save();print('SCREEN',json.dumps(summary),flush=True)
 jobs=defaultdict(list)
 for e in examples:
  for mode in cfg['control_modes']:jobs[('mlp',len(e['ids']))].append((e,mode))
  for mode in ('shared','output_bias'):jobs[('final',len(e['ids']))].append((e,mode))
 for (site,length),ee in sorted(jobs.items()):
  for begin in range(0,len(ee),12):
   batch=ee[begin:begin+12];vv=torch.stack([vectors[e['row']['family']][mode]*(1 if e['variant']==0 else -1) for e,mode in batch]);ids=torch.tensor([e['ids'] for e,mode in batch],device='cuda');logits=run(model,ids,vv,site)
   for ix,(e,mode) in enumerate(batch):
    r=e['row'];variant=e['variant'];key=(r['id'],variant);mm=measure(logits[ix],e['groups']);delta=mm['margin']-native[key]['margin'];target=r['expected'][1-variant];lp=logits[ix].log_softmax(0).cpu();oldlp=logps[key];kl=float((oldlp.exp()*(oldlp-lp)).sum());entropy=float(-(lp.exp()*lp).sum());oldentropy=float(-(oldlp.exp()*oldlp).sum())
    result['interventions'].append(dict(id=r['id'],family=r['family'],readout=r['readout'],kind=r['kind'],variant=variant,site=site,mode=mode,margin=mm['margin'],raw_margin_delta=delta,donor_directed_delta=delta*(1 if target==0 else -1) if r['kind']=='task' else None,KL=kl,entropy_change=entropy-oldentropy,top1=mm['top1']))
  save();print('done',site,length,flush=True)
 result['audits']={'self_max_margin_delta':max(abs(c['raw_margin_delta']) for c in result['interventions'] if c['mode']=='identity'),'self_max_abs_KL':max(abs(c['KL']) for c in result['interventions'] if c['mode']=='identity'),'finite':all(np.isfinite(c['KL']) and np.isfinite(c['margin']) for c in result['interventions']),'manifest_unchanged':mh==hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
