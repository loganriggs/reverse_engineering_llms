#!/usr/bin/env python3
"""H545: all three-name permutations, balanced order, frozen reader/source interventions."""
import json,time,hashlib,itertools
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id
from run_bilin18_reader_operation_format import run,HEADS
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_three_name_mapping';torch.set_num_threads(6)
def score(z,ids,answer,query):
 c=z[ids];other=[i for i in range(3) if i!=answer];rival=next((i for i in range(3) if i not in (answer,query)),None) if answer!=query else None
 return {'candidate_logits':c.tolist(),'candidate_correct':int(c.argmax())==answer,'full_top1':int(z.argmax())==ids[answer],'target_rank':int((z>z[ids[answer]]).sum())+1,'target_probability':float(z.softmax(0)[ids[answer]]),'correct_margin':float(c[answer]-torch.logsumexp(c[other],0)),'nonquery_margin':float(c[answer]-c[rival]) if rival is not None else None,'nonquery_correct':bool(c[answer]>c[rival]) if rival is not None else None,'predicted_token':int(z.argmax())}
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_reader_mapping_source_roles.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 triples=[('Arthur','Olivia','Frank'),('Albert','Sophia','Eric'),('Fred','Maria','Adam'),('Philip','Nicole','Oscar'),('Richard','Amy','Simon'),('Steven','Karen','Alex'),('Tony','Julie','Victor'),('Vincent','Angela','Andrew')];rows=[]
 for item,names in enumerate(triples):
  for perm in itertools.permutations(range(3)):
   fixed=sum(perm[i]==i for i in range(3));kind='identity' if fixed==3 else 'transposition' if fixed==1 else 'cycle';pname=''.join(map(str,perm))
   for rotation in range(3):
    order=[(rotation+i)%3 for i in range(3)];text='';vp={}
    for key in order:
     stem=text+f'from {names[key]}, to';vp[key]=len(tok.encode(stem,add_special_tokens=False));text=stem+f' {names[perm[key]]}; '
    for query in range(3):
     prompt=text+f'from {names[query]}, to';ids=tok.encode(prompt,add_special_tokens=False);candidate_ids=[continuation_id(tok,prompt,' '+n) for n in names];assert all(x is not None for x in candidate_ids);qp=len(tok.encode(text+'from',add_special_tokens=False));assert ids[qp]==candidate_ids[query];assert all(ids[vp[k]]==candidate_ids[perm[k]] for k in range(3))
     rows.append(dict(id=f'tri-{item}-{pname}-{rotation}-{query}',block=f'{item}-{pname}-{rotation}',item=item,names=list(names),permutation=list(perm),kind=kind,rotation=rotation,order=order,query=query,answer=perm[query],fixed_point=perm[query]==query,prompt=prompt,ids=ids,candidate_ids=candidate_ids,query_position=qp,value_positions=[vp[k] for k in range(3)]))
 cfg={'experiment':'H545','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'heads':HEADS,'design':'8 name triples x all 6 permutations x 3 cyclic entry orders x all 3 queries = 432 prompts. Positions and token counts fixed within blocks; every key occupies each entry position across order rotations.','competence':'Report three-way candidate accuracy, full-vocabulary top1, all-three-query block accuracy and nonquery-rival accuracy for nonfixed queries. Strong gate requires >=75% all-three-query blocks and >=50% full-top1 per permutation kind. No individual-case filtering.','sites':['query','value_of_key0','value_of_key1','value_of_key2'],'intervention':'Replace one source-to-head bus contribution at final destination with mean of the three name-value vectors. Other paths and downstream computation live. Self adds zero.','hypotheses':['Three-way success requires evidence separating two nonquery alternatives, beyond suppressing the query.','The frozen heads preferentially copy the correct value if they select an association.','Broad copying with query correction predicts weak correct-value preference and possible failure on cycles or fixed points.'],'scope':'No head fitting or selection. Individual head/source neutralizations, not exhaustive ablations. Known names in new combinations, not training-data holdout.','precision':'float32 compute after bfloat16-rounded loader'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H
 def token_bus(token):
  x0=F.rms_norm(model.embedding(torch.tensor([token],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;return model.blocks[0].attn.c_v(F.rms_norm(x,(model.config.n_embd,))).reshape(H,D)
 cache={};native={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for length,rr in sorted(buckets.items()):
  for begin in range(0,len(rr),24):
   batch=rr[begin:begin+24];logits,cc=run(model,torch.tensor([r['ids'] for r in batch],device='cuda'),capture=True)
   for ix,r in enumerate(batch):cache[r['id']]={n:v[ix].clone() for n,v in cc.items()};native[r['id']]=score(logits[ix],r['candidate_ids'],r['answer'],r['query'])
 result={'config':cfg,'manifest_sha256':mh,'native':[{'id':r['id'],'item':r['item'],'kind':r['kind'],'rotation':r['rotation'],'query':r['query'],'answer':r['answer'],'fixed_point':r['fixed_point'],**native[r['id']]} for r in rows],'competence':{},'geometry':[],'interventions':[],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 for kind in ('identity','transposition','cycle'):
  rr=[r for r in rows if r['kind']==kind];nn=[native[r['id']] for r in rr];blocks=sorted(set(r['block'] for r in rr));complete=[all(native[r['id']]['candidate_correct'] for r in rr if r['block']==b) for b in blocks];nq=[n['nonquery_correct'] for n in nn if n['nonquery_correct'] is not None];s={'prompts':len(rr),'candidate_accuracy':float(np.mean([n['candidate_correct'] for n in nn])),'full_vocab_top1':float(np.mean([n['full_top1'] for n in nn])),'all_queries_block_accuracy':float(np.mean(complete)),'nonquery_rival_accuracy':float(np.mean(nq)) if nq else None};s['strong_gate']=s['all_queries_block_accuracy']>=.75 and s['full_vocab_top1']>=.5;result['competence'][kind]=s
 save();print('SCREEN',result['competence'],flush=True)
 # Cache native state for follow-up without retaining intervention graphs.
 torch.save({'cache':cache,'rows':rows},ART/f'{STEM}_native_cache.pt')
 jobs=defaultdict(list);means={}
 for r in rows:
  if r['item'] not in means:means[r['item']]=torch.stack([token_bus(t) for t in r['candidate_ids']]).mean(0)
  c=cache[r['id']];T=len(r['ids']);sites=[('query',r['query_position'],None)]+[(f'value_{k}',p,k) for k,p in enumerate(r['value_positions'])]
  for l,h in HEADS:
   attn=model.blocks[l].attn;q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q);q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin);f1=torch.einsum('bhd,bthd->bht',q[:,-1],k)/D;f2=torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/D
   for site,p,keyidx in sites:
    coef=f1[0,h,p]*f2[0,h,p]*attn.lamb;role='query' if keyidx is None else 'correct_value' if keyidx==r['query'] else 'competing_value';meta={'id':r['id'],'item':r['item'],'kind':r['kind'],'fixed_point':r['fixed_point'],'query':r['query'],'answer':r['answer'],'rotation':r['rotation'],'layer':l,'head':h,'site':site,'role':role,'source_position':p,'source_identity':r['query'] if keyidx is None else r['permutation'][keyidx]};result['geometry'].append(meta|{'coefficient':float(coef),'factor1':float(f1[0,h,p]),'factor2':float(f2[0,h,p])});delta=coef*F.linear(means[r['item']][h]-c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);jobs[(l,T)].append((r,meta,delta.cpu(),'neutralize'))
  meta={'id':r['id'],'item':r['item'],'kind':r['kind'],'fixed_point':r['fixed_point'],'query':r['query'],'answer':r['answer'],'rotation':r['rotation'],'layer':8,'head':3,'site':'query','role':'query','source_position':r['query_position'],'source_identity':r['query']};jobs[(8,T)].append((r,meta,torch.zeros(model.config.n_embd),'self'))
 for (l,T),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),24):
   batch=jj[begin:begin+24];logits,_=run(model,torch.tensor([r['ids'] for r,m,d,mode in batch],device='cuda'),patch=(l,torch.stack([d for r,m,d,mode in batch])))
   for ix,(r,meta,d,mode) in enumerate(batch):
    ss=score(logits[ix],r['candidate_ids'],r['answer'],r['query']);nn=native[r['id']];result['interventions'].append(meta|{'mode':mode,**ss,'correct_margin_delta':ss['correct_margin']-nn['correct_margin'],'nonquery_margin_delta':ss['nonquery_margin']-nn['nonquery_margin'] if ss['nonquery_margin'] is not None else None})
   if begin%480==0:save();print('progress',l,begin,'/',len(jj),flush=True)
  save()
 result['audits']={'self_max_margin_delta':max(abs(c['correct_margin_delta']) for c in result['interventions'] if c['mode']=='self'),'self_max_nonquery_margin_delta':max(abs(c['nonquery_margin_delta']) for c in result['interventions'] if c['mode']=='self' and c['nonquery_margin_delta'] is not None),'source_tokens_and_positions_asserted':True,'finite':all(np.isfinite(c['correct_margin']) for c in result['interventions']),'manifest_unchanged':mh==hashlib.sha256(mp.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
