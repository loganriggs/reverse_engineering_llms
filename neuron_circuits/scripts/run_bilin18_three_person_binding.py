#!/usr/bin/env python3
"""H546: natural three-person binding with irrelevant-person position controls."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id
from run_bilin18_three_name_mapping import score
from run_bilin18_reader_operation_format import run,HEADS
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_three_person_binding';torch.set_num_threads(6)
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_three_name_mapping.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);triples=[('Arthur','Olivia','Frank'),('Albert','Sophia','Eric'),('Fred','Maria','Adam'),('Philip','Nicole','Oscar'),('Richard','Amy','Simon'),('Steven','Karen','Alex'),('Tony','Julie','Victor'),('Vincent','Angela','Andrew')];rows=[]
 for item,names in enumerate(triples):
  for rotation in range(3):
   a,b,c=[(rotation+i)%3 for i in range(3)]
   for fmt in ('distractor_first','distractor_middle','distractor_parenthetical'):
    for giver,recipient in ((a,b),(b,a)):
     text='';positions={}
     def lit(s):
      nonlocal text
      text+=s
     def name(role,i):
      nonlocal text
      tid=continuation_id(tok,text,' '+names[i]);assert tid is not None;positions[role]=len(tok.encode(text,add_special_tokens=False));text+=' '+names[i]
     if fmt=='distractor_first':lit('Meanwhile,');name('distractor',c);lit(' stayed home. ')
     lit('Only');name('group_a',a);lit(' and');name('group_b',b);lit(' went to the market.')
     if fmt=='distractor_middle':lit(' Meanwhile,');name('distractor',c);lit(' stayed home.')
     name('repeated_giver',giver)
     if fmt=='distractor_parenthetical':lit(', while');name('distractor',c);lit(' stayed home,')
     lit(' gave a parcel to');ids=tok.encode(text,add_special_tokens=False);candidates=[continuation_id(tok,text,' '+n) for n in names];assert all(x is not None for x in candidates)
     sites={'initial_giver':positions['group_a' if giver==a else 'group_b'],'recipient':positions['group_b' if giver==a else 'group_a'],'distractor':positions['distractor'],'repeated_giver':positions['repeated_giver']}
     assert all(ids[p]==candidates[recipient if site=='recipient' else c if site=='distractor' else giver] for site,p in sites.items())
     rows.append(dict(id=f'bind3-{item}-{rotation}-{fmt}-{giver}',block=f'{item}-{rotation}-{fmt}',item=item,names=list(names),rotation=rotation,format=fmt,query=giver,answer=recipient,distractor=c,prompt=text,ids=ids,candidate_ids=candidates,sites=sites))
 cfg={'experiment':'H546','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'heads':HEADS,'design':'Eight triples, three role rotations, two givers, three distractor placements =144 prompts. Only two people go to market; third stays home. Giver changes within each paired block.','competence':'Three-way and full-vocabulary accuracy, recipient-vs-distractor accuracy excluding giver, and both-giver block accuracy. Strong gate >=75% paired blocks and >=50% full-top1 per format.','intervention':'Neutralize one named source/head contribution to mean of the three name-value vectors, then run downstream normally. Same frozen four heads, no reselection.','hypotheses':['Natural binding distinguishes recipient from an irrelevant third person where arbitrary mapping is unreliable.','An association-selective reader should favor recipient over distractor even when distractor is more recent.','Giver suppression alone cannot determine the recipient among two nongiver candidates.'],'scope':'Declared natural-language expectation and model competence, not a logical guarantee that a parcel could never reach an absent person. Source/head-local effects, not whole-circuit ablation.','precision':'float32 compute after bfloat16-rounded loader'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H
 def token_bus(t):
  x0=F.rms_norm(model.embedding(torch.tensor([t],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;return model.blocks[0].attn.c_v(F.rms_norm(x,(model.config.n_embd,))).reshape(H,D)
 cache={};native={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for length,rr in sorted(buckets.items()):
  for begin in range(0,len(rr),24):
   batch=rr[begin:begin+24];logits,cc=run(model,torch.tensor([r['ids'] for r in batch],device='cuda'),capture=True)
   for ix,r in enumerate(batch):cache[r['id']]={n:v[ix].clone() for n,v in cc.items()};native[r['id']]=score(logits[ix],r['candidate_ids'],r['answer'],r['query'])
 result={'config':cfg,'manifest_sha256':mh,'native':[dict(id=r['id'],item=r['item'],rotation=r['rotation'],format=r['format'],**native[r['id']]) for r in rows],'competence':{},'geometry':[],'interventions':[],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 for fmt in sorted(set(r['format'] for r in rows)):
  rr=[r for r in rows if r['format']==fmt];nn=[native[r['id']] for r in rr];blocks=set(r['block'] for r in rr);s={'prompts':len(rr),'candidate_accuracy':float(np.mean([n['candidate_correct'] for n in nn])),'full_vocab_top1':float(np.mean([n['full_top1'] for n in nn])),'recipient_vs_distractor_accuracy':float(np.mean([n['nonquery_correct'] for n in nn])),'both_givers_block_accuracy':float(np.mean([all(native[r['id']]['candidate_correct'] for r in rr if r['block']==b) for b in blocks]))};s['strong_gate']=s['both_givers_block_accuracy']>=.75 and s['full_vocab_top1']>=.5;result['competence'][fmt]=s
 save();print('SCREEN',result['competence'],flush=True);torch.save({'cache':cache,'rows':rows},ART/f'{STEM}_native_cache.pt');means={};jobs=defaultdict(list)
 for r in rows:
  if r['item'] not in means:means[r['item']]=torch.stack([token_bus(t) for t in r['candidate_ids']]).mean(0)
  c=cache[r['id']];T=len(r['ids'])
  for l,h in HEADS:
   attn=model.blocks[l].attn;q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q);q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin);f1=torch.einsum('bhd,bthd->bht',q[:,-1],k)/D;f2=torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/D
   for site,p in r['sites'].items():
    coef=f1[0,h,p]*f2[0,h,p]*attn.lamb;meta={k:r[k] for k in ('id','item','rotation','format')}|{'layer':l,'head':h,'site':site,'source_position':p};result['geometry'].append(meta|{'coefficient':float(coef),'factor1':float(f1[0,h,p]),'factor2':float(f2[0,h,p])});delta=coef*F.linear(means[r['item']][h]-c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);jobs[(l,T)].append((r,meta,delta.cpu(),'neutralize'))
  meta={k:r[k] for k in ('id','item','rotation','format')}|{'layer':8,'head':3,'site':'repeated_giver','source_position':r['sites']['repeated_giver']};jobs[(8,T)].append((r,meta,torch.zeros(model.config.n_embd),'self'))
 for (l,T),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),24):
   batch=jj[begin:begin+24];logits,_=run(model,torch.tensor([r['ids'] for r,m,d,mode in batch],device='cuda'),patch=(l,torch.stack([d for r,m,d,mode in batch])))
   for ix,(r,meta,d,mode) in enumerate(batch):
    ss=score(logits[ix],r['candidate_ids'],r['answer'],r['query']);nn=native[r['id']];result['interventions'].append(meta|{'mode':mode,**ss,'correct_margin_delta':ss['correct_margin']-nn['correct_margin'],'nonquery_margin_delta':ss['nonquery_margin']-nn['nonquery_margin']})
  save();print('done',l,T,flush=True)
 result['audits']={'self_max_margin_delta':max(abs(c['correct_margin_delta']) for c in result['interventions'] if c['mode']=='self'),'source_tokens_and_roles_asserted':True,'finite':all(np.isfinite(c['correct_margin']) for c in result['interventions']),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),'manifest_unchanged':mh==hashlib.sha256(mp.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
