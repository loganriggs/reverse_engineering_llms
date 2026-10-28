"""H747: locked name x wording confirmation of frozen paired-edit bases."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_static_bus_map import static_bus
from run_bilin18_query_key_edit_subspaces import rotate,projected_edit
from run_bilin18_recall_query_key_cross import coefficient
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_fresh_wording_subspace_effects'
def make_panel(tok):
 sets={'old':[('Arthur','Olivia'),('Albert','Sophia'),('Fred','Maria'),('Philip','Nicole'),('Richard','Amy'),('Steven','Karen'),('Tony','Julie'),('Vincent','Angela')],'new':[('Louis','Martha'),('Alan','Betty'),('Edwin','Paula'),('Harold','Doris'),('Frank','Anita'),('Roger','Elena'),('Walter','Fiona'),('Martin','Heidi')]};rows=[]
 for names,items in sets.items():
  for wording in ['old','new']:
   for fmt in ['narrative','list']:
    for i,(a,b) in enumerate(items):
     if fmt=='narrative':
      if wording=='old':binding=[f'{a} and {b} met at the concert. {g} passed a message to' for g in [a,b]];copy=[f'He talked to {v}. He talked to {v}. He talked to' for v in [a,b]]
      else:binding=[f'{a} and {b} were at the library. {g} sent a book to' for g in [a,b]];copy=[f'She wrote to {v}. She wrote to {v}. She wrote to' for v in [a,b]]
     else:
      if wording=='old':binding=[f'giver {a}, receiver {b}; giver {b}, receiver {a}; giver {g}, receiver' for g in [a,b]];copy=[f'receiver {v}; receiver {v}; receiver' for v in [a,b]]
      else:binding=[f'source {a}, destination {b}; source {b}, destination {a}; source {g}, destination' for g in [a,b]];copy=[f'destination {v}; destination {v}; destination' for v in [a,b]]
     length=len(tok.encode(binding[0],add_special_tokens=False));padded=[]
     for text in copy:
      candidates=['Note '*n+text for n in range(80)];ok=[t for t in candidates if len(tok.encode(t,add_special_tokens=False))==length];padded.append(ok[0] if ok else text)
     for family,prompts,answers in [('entity_binding',binding,[b,a]),('sequence_copy',padded,[a,b])]:rows.append(dict(id=f'h747-{names}-{wording}-{fmt}-{family}-{i}',name_set=names,wording=wording,format=fmt,family=family,item=i,prompts=prompts,answers=answers))
 return rows
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=make_panel(tok);manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));paths={'manifest':manifest,'bases':ART/'bilin18_operation_nuisance_subspaces_captures.pt','old_geometry':ART/'bilin18_query_key_edit_subspaces.json','old_manifest':ART/'bilin18_reader_operation_format_manifest.json','model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'projection':ROOT/'scripts/run_bilin18_query_key_edit_subspaces.py','static_bus':ROOT/'scripts/run_bilin18_static_bus_map.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};bases=torch.load(paths['bases'],weights_only=False)['bases'];cfg=dict(experiment='H747',hashes=hashes,design='Fresh wording confirmation. Names reused from H744; wording tags old/new mean fresh template A/B in this experiment, not previous wording. Two narrative and two list templates locked before outputs. Frozen H746 original, new-wording, pooled-operation, identity and wording rank4 bases plus three random controls. Receiver-norm-preserving, separate q2/k2. Full factor2/native/zero controls. Single-edge original-value and absent Kevin/Brian downstream interventions. No fitting or candidate selection.',competence_gate='>=4 valid pairs, >=.75 both endpoint choices, >=.5 intended full-vocabulary top1; both operation cells must pass. All token-valid contexts retained. Exact source/final token and position matching required.',limitations='New wording relative to H742/H744/H746; reused names, not independent name confirmation. Several lexical cues change jointly. Surgical single edge, not global Q/K replacement. No natural reserved-test evaluation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));examples={};excluded=[];lookup={r['id']:r for r in rows}
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,' '+a)] for a in r['answers']];ee.append(dict(ids=tok.encode(text,add_special_tokens=False),groups=gs))
  if any(g[0] is None for e in ee for g in e['groups']) or len(ee[0]['ids'])!=len(ee[1]['ids']):excluded.append(dict(id=r['id'],reason='answer token validity or endpoint length'));continue
  pos=[i for i,(a,b) in enumerate(zip(ee[0]['ids'],ee[1]['ids'])) if a!=b]
  if not pos:excluded.append(dict(id=r['id'],reason='no changed source'));continue
  for v,e in enumerate(ee):e['source_position']=pos[-1];examples[r['id'],v]=e
 pairs=[];matching=[]
 for r in rows:
  if r['family']!='entity_binding':continue
  other=r['id'].replace('entity_binding','sequence_copy')
  for v in [0,1]:
   ak,bk=(r['id'],v),(other,v)
   if ak not in examples or bk not in examples:continue
   a,b=examples[ak],examples[bk];m=dict(name_set=r['name_set'],wording=r['wording'],format=r['format'],item=r['item'],variant=v,final_position_match=len(a['ids'])==len(b['ids']),source_position_match=a['source_position']==b['source_position'],source_token_match=a['ids'][a['source_position']]==b['ids'][b['source_position']],final_token_match=a['ids'][-1]==b['ids'][-1]);m['pass']=all(m[k] for k in ['final_position_match','source_position_match','source_token_match','final_token_match']);matching.append(m)
   if m['pass']:pairs.extend([(ak,bk),(bk,ak)])
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 attn=model.blocks[13].attn;lam=float(attn.lamb);native={};states={};buckets=defaultdict(list);headerrors=[];forwards=0
 for key,e in examples.items():buckets[len(e['ids'])].append(key)
 for length,keys in buckets.items():
  for lo in range(0,len(keys),12):
   bb=keys[lo:lo+12];cap={};hooks=[]
   def pre(m,args):cap['bus']=args[1].view(len(bb),length,9,128)[:,:,0].cpu()
   hooks.append(attn.register_forward_pre_hook(pre))
   for name in ['c_q','c_k','c_q2','c_k2','c_v']:
    def h(m,args,out,name=name):cap[name]=out[:,:,:128].cpu()
    hooks.append(getattr(attn,name).register_forward_hook(h))
   def ph(m,args):cap['head']=args[0][:,-1,:128].cpu()
   hooks.append(attn.c_proj.register_forward_pre_hook(ph))
   try:logits,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hooks:h.remove()
   forwards+=len(bb);raw=[cap[n].cuda()[:,:,None,:] for n in ['c_q','c_k','c_q2','c_k2']];norm=[F.rms_norm(t,(128,)) for t in raw];cos,sin=attn.rotary(raw[0]);q,k,q2,k2=[model.tt.apply_rotary_emb(t,cos,sin) for t in norm];pat=((q[:,-1]*k[:,:,0]).sum(-1)/128)*((q2[:,-1]*k2[:,:,0]).sum(-1)/128);V=(1-lam)*cap['c_v'].cuda()+lam*cap['bus'].cuda();headrec=torch.einsum('bt,btd->bd',pat,V);headerrors.extend(((headrec-cap['head'].cuda()).norm(dim=1)/cap['head'].cuda().norm(dim=1).clamp_min(1e-12)).tolist())
   for n,key in enumerate(bb):
    pos=examples[key]['source_position'];native[key]=logits[n].cpu();states[key]=dict(q1=q[n,-1,0].cpu().double(),k1=k[n,pos,0].cpu().double(),q2=q2[n,-1,0].cpu().double(),k2=k2[n,pos,0].cpu().double(),uq2=norm[2][n,-1,0].cpu().double(),uk2=norm[3][n,pos,0].cpu().double())
 competence={}
 for names in ['old','new']:
  for wording in ['old','new']:
   for fmt in ['narrative','list']:
    for fam in ['entity_binding','sequence_copy']:
     rr=[r for r in rows if (r['name_set'],r['wording'],r['format'],r['family'])==(names,wording,fmt,fam) and (r['id'],0) in examples];both=[];top=[]
     for r in rr:
      mm=[measure(native[r['id'],v],examples[r['id'],v]['groups']) for v in [0,1]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend(mm[v]['top1'][v] for v in [0,1])
     s=dict(pairs=len(rr),both_endpoint_choice=float(np.mean(both)) if rr else 0,full_vocab_top1=float(np.mean(top)) if rr else 0);s['pass']=s['pairs']>=4 and s['both_endpoint_choice']>=.75 and s['full_vocab_top1']>=.5;competence['|'.join([names,wording,fmt,fam])]=s
 print('COMPETENCE',competence,flush=True);geometry=[];jobs=[];variants=[(s,None) for s in ['frozen_H742','operation_new','operation_pooled','identity','wording','random_7421','random_7422','random_7423','full','native','zero']]
 for key,donor in pairs:
  meta=lookup[key[0]];e=examples[key];r=states[key];d=states[donor];a=float(coefficient(r,lam));full=float(coefficient({**r,'q2':d['q2'],'k2':d['k2']},lam));src=e['ids'][e['source_position']];Bsrc=static_bus(model,[src])[0];tokens=[continuation_id(tok,meta['prompts'][key[1]],word) for word in [' Kevin',' Brian']];assert all(t is not None and t not in e['ids'] for t in tokens);B=static_bus(model,tokens)
  for scope,seed in variants:
   if scope not in ['full','native','zero']:
    qq=projected_edit(r['uq2'],d['uq2'],bases[scope,'q2'],True);kk=projected_edit(r['uk2'],d['uk2'],bases[scope,'k2'],True);pred=float(coefficient({**r,'q2':rotate(qq,len(e['ids'])-1),'k2':rotate(kk,e['source_position'])},lam))
   else:pred={'full':full,'native':a,'zero':0.}[scope]
   rec=dict(id=key[0],receiver=key[1],name_set=meta['name_set'],wording=meta['wording'],format=meta['format'],family=meta['family'],item=meta['item'],scope=scope,seed=seed,native_alpha=a,target_alpha=full,predicted_alpha=pred)

   trial=len(geometry);geometry.append(rec)
   for mode,v,delta in [('native_value',None,(pred-a)*Bsrc),('absent',0,pred*B[0]-a*Bsrc),('absent',1,pred*B[1]-a*Bsrc)]:jobs.append(dict(trial=trial,key=key,mode=mode,variant=v,tokens=tokens,delta=delta.cpu()))
 results={};effects={};buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];delta=torch.stack([r['delta'] for r in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,:128]+=delta.to(y);return(y,)
   hook=attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[r['key']]['ids'] for r in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for n,job in enumerate(bb):
    key=job['key']
    if job['mode']=='absent':x,y=job['tokens'];results[job['trial'],job['variant']]=float(logits[n,x]-logits[n,y])
    else:gs=examples[key]['groups'];change=measure(logits[n],gs)['margin']-measure(native[key],gs)['margin'];effects[job['trial']]=change*(1 if key[1]==0 else -1)
 records=[dict(**r,absent_identity_effect=results[t,0]-results[t,1],native_value_intended_margin_change=effects[t]) for t,r in enumerate(geometry)];identity=max(abs(r['native_value_intended_margin_change']) for r in records if r['scope']=='native');zero=max(abs(r['absent_identity_effect']) for r in records if r['scope']=='zero');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},native_head_factorization_max_relative=max(headerrors),native_value_identity_max_abs=identity,zero_contrast_max_abs=zero,finite=all(np.isfinite(r['absent_identity_effect']) for r in records),whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(headerrors)<1e-4 and identity<1e-4 and zero<1e-4;torch.save(dict(native_logits=native,states=states,examples=examples,pairs=pairs),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,excluded=excluded,matching=matching,competence=competence,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
