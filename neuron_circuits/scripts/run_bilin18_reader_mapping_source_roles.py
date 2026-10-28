#!/usr/bin/env python3
"""H544: identity versus swap mappings separate source-role effects from answer steering."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id,measure
from run_bilin18_reader_operation_format import run,HEADS
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_reader_mapping_source_roles';torch.set_num_threads(6)
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_reader_operation_format.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);names=[('Arthur','Olivia'),('Albert','Sophia'),('Fred','Maria'),('Philip','Nicole'),('Richard','Amy'),('Steven','Karen'),('Tony','Julie'),('Vincent','Angela')];thirds=[('Kevin','Brian'),('Ben','Zoe'),('Charles','Nancy'),('Edward','Helen'),('Harry','Betty'),('Louis','Anne'),('Martin','Joan'),('Walter','Carol')];rows=[]
 for i,(a,b) in enumerate(names):
  for mapping in ('identity','swap'):
   targets=(a,b) if mapping=='identity' else (b,a)
   for order in ('ab','ba'):
    entries=[(a,targets[0]),(b,targets[1])]
    if order=='ba':entries=entries[::-1]
    text='';vp={}
    for key,value in entries:
     stem=text+f'from {key}, to';vp[key]=len(tok.encode(stem,add_special_tokens=False));text=stem+f' {value}; '
    prompts=[text+f'from {q}, to' for q in (a,b)]
    rows.append(dict(id=f'map-{mapping}-{order}-{i}',item=i,mapping=mapping,order=order,prompts=prompts,answers=list(targets),names=[a,b],third=list(thirds[i]),value_positions=[vp[a],vp[b]]))
 cfg={'experiment':'H544','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'heads':HEADS,'design':'Identity A->A,B->B versus swap A->B,B->A, each with both entry orders and both query names. Same name frequencies and local query syntax; record exact positions.','sites':['query','mapped_value','other_value'],'interventions':['midpoint of A/B bus values','third0','third1','self'],'hypotheses':['A source-role query suppressor stays suppressive under identity mappings even though query identity is then correct.','A mapped-value reader copies the demonstrated target across identity/swap mappings.','Midpoint neutralization harms correct mapped-value evidence, while neutralizing query suppression can help identity mappings and hurt swap mappings.'],'scope':'One source position, one head, final attention destination; other paths and downstream computation live. This is not whole-circuit ablation.','selection':'All eight name pairs, four mapping/order cells, all heads/sites, no fits or postselection. Native competence reported first.','precision':'float32 computation after shared-loader bfloat16 rounding'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),'pairs',flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H
 def token_bus(token):
  x0=F.rms_norm(model.embedding(torch.tensor([token],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;return model.blocks[0].attn.c_v(F.rms_norm(x,(model.config.n_embd,))).reshape(H,D)
 examples={};cache={};native={};buckets=defaultdict(list)
 for r in rows:
  encoded=[tok.encode(t,add_special_tokens=False) for t in r['prompts']];assert len(encoded[0])==len(encoded[1]);changes=[p for p,(a,b) in enumerate(zip(*encoded)) if a!=b];assert len(changes)==1
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,' '+a)] for a in r['answers']];ts=[[continuation_id(tok,text,' '+a)] for a in r['third']];ns=[continuation_id(tok,text,' '+a) for a in r['names']];assert all(g[0] is not None for g in gs+ts) and all(t is not None for t in ns);key=(r['id'],v);sites={'query':changes[0],'mapped_value':r['value_positions'][v],'other_value':r['value_positions'][1-v]};assert encoded[v][sites['mapped_value']]==gs[v][0] and encoded[v][sites['query']]==ns[v];examples[key]={'ids':encoded[v],'groups':gs,'third':ts,'names':ns,'sites':sites};buckets[len(encoded[v])].append(key)
 for length,keys in sorted(buckets.items()):
  for begin in range(0,len(keys),8):
   kk=keys[begin:begin+8];logits,cc=run(model,torch.tensor([examples[k]['ids'] for k in kk],device='cuda'),capture=True)
   for ix,key in enumerate(kk):cache[key]={n:v[ix].clone() for n,v in cc.items()};native[key]=measure(logits[ix],examples[key]['groups'])
 result={'config':cfg,'manifest_sha256':mh,'native':[dict(id=k[0],variant=k[1],**v) for k,v in native.items()],'competence':{},'geometry':[],'interventions':[],'third_comparisons':[],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 for mapping in ('identity','swap'):
  for order in ('ab','ba'):
   rr=[r for r in rows if r['mapping']==mapping and r['order']==order];both=[native[(r['id'],0)]['margin']>0 and native[(r['id'],1)]['margin']<0 for r in rr];top=[native[(r['id'],v)]['top1'][v] for r in rr for v in (0,1)];result['competence'][mapping+'_'+order]={'both_endpoint_choice':float(np.mean(both)),'full_vocab_top1':float(np.mean(top)),'pass':float(np.mean(both))>=.75 and float(np.mean(top))>=.5}
 print('SCREEN',result['competence'],flush=True);jobs=defaultdict(list)
 for r in rows:
  for v in (0,1):
   key=(r['id'],v);e=examples[key];c=cache[key];T=len(e['ids']);mid=(token_bus(e['names'][0])+token_bus(e['names'][1]))/2;third=[token_bus(g[0]) for g in e['third']]
   for l,h in HEADS:
    attn=model.blocks[l].attn;q,k,q2,k2=[c[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_k','c_q2','c_k2')];cos,sin=attn.rotary(q);q,k=model.tt.apply_rotary_emb(F.rms_norm(q,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k,(D,)),cos,sin);q2,k2=model.tt.apply_rotary_emb(F.rms_norm(q2,(D,)),cos,sin),model.tt.apply_rotary_emb(F.rms_norm(k2,(D,)),cos,sin);f1=torch.einsum('bhd,bthd->bht',q[:,-1],k)/D;f2=torch.einsum('bhd,bthd->bht',q2[:,-1],k2)/D
    for site,p in e['sites'].items():
     coef=f1[0,h,p]*f2[0,h,p]*attn.lamb;result['geometry'].append(dict(id=r['id'],item=r['item'],mapping=r['mapping'],order=r['order'],variant=v,layer=l,head=h,site=site,position=p,final_position=T-1,source_token=e['ids'][p],coefficient=float(coef),factor1=float(f1[0,h,p]),factor2=float(f2[0,h,p])))
     for mode,value in [('midpoint',mid),('third0',third[0]),('third1',third[1])]:
      delta=coef*F.linear(value[h]-c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]);jobs[(l,T)].append((r,v,l,h,site,mode,delta.cpu()))
   jobs[(8,T)].append((r,v,8,3,'query','self',torch.zeros(model.config.n_embd)))
 save()
 for (l,T),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),8):
   batch=jj[begin:begin+8];ids=torch.tensor([examples[(r['id'],v)]['ids'] for r,v,l,h,site,mode,d in batch],device='cuda');logits,_=run(model,ids,patch=(l,torch.stack([d for r,v,l,h,site,mode,d in batch])))
   for ix,(r,v,l,h,site,mode,d) in enumerate(batch):
    key=(r['id'],v);m=measure(logits[ix],examples[key]['groups']);t=measure(logits[ix],examples[key]['third']);result['interventions'].append(dict(id=r['id'],item=r['item'],mapping=r['mapping'],order=r['order'],variant=v,layer=l,head=h,site=site,mode=mode,correct_margin_delta=(m['margin']-native[key]['margin'])*(1 if v==0 else -1),third_margin=t['margin']))
  save()
 ix={(c['id'],c['variant'],c['layer'],c['head'],c['site'],c['mode']):c for c in result['interventions']}
 for g in result['geometry']:
  key=tuple(g[k] for k in ('id','variant','layer','head','site'));effect=ix[key+('third0',)]['third_margin']-ix[key+('third1',)]['third_margin'];result['third_comparisons'].append({k:g[k] for k in ('id','item','mapping','order','variant','layer','head','site','coefficient')}|{'identity_effect':effect})
 result['audits']={'source_positions_and_tokens_asserted':True,'self_max_margin_delta':max(abs(c['correct_margin_delta']) for c in result['interventions'] if c['mode']=='self'),'finite':all(np.isfinite(c['correct_margin_delta']) and np.isfinite(c['third_margin']) for c in result['interventions']),'manifest_unchanged':mh==hashlib.sha256(mp.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()
