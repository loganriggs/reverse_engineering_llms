#!/usr/bin/env python3
"""H548: position-matched cue timing, frozen distractor deletion, actual Q/K swaps."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,continuation_id
from run_bilin18_three_name_mapping import score
from run_bilin18_reader_operation_format import run as native_run
from run_bilin18_reader_query_key_context import run as swapped_run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_distractor_cue_availability';torch.set_num_threads(6)
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_distractor_rescue_controls.json';tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);triples=[('John','Mary','Peter'),('Alice','Bob','Jane'),('Carol','Dave','Tom'),('Sarah','James','Anna'),('David','Kate','Mark'),('Lucy','Paul','Lisa'),('George','Susan','Michael'),('Robert','Emily','Daniel')];rows=[]
 clauses={'home':{'before':('At home today,',' quietly waited.'),'after':('Earlier today though,',' stayed home.')},'work':{'before':('At work today,',' just quietly waited.'),'after':('Earlier today though,',' stayed at work.')}}
 for item,names in enumerate(triples):
  for rotation in range(3):
   a,b,c=[(rotation+i)%3 for i in range(3)]
   for giver,recipient in ((a,b),(b,a)):
    for cue,forms in clauses.items():
     for timing,(prefix,suffix) in forms.items():
      text=prefix;pos=len(tok.encode(text,add_special_tokens=False));assert continuation_id(tok,text,' '+names[c]) is not None;text+=' '+names[c]+suffix+f' Only {names[a]} and {names[b]} went to the market. {names[giver]} gave a parcel to';ids=tok.encode(text,add_special_tokens=False);cand=[continuation_id(tok,text,' '+n) for n in names];assert all(t is not None for t in cand) and ids[pos]==cand[c];pair=f'{item}-{rotation}-{giver}-{cue}';rows.append(dict(id=pair+'-'+timing,pair=pair,item=item,rotation=rotation,cue=cue,timing=timing,names=list(names),query=giver,answer=recipient,distractor=c,prompt=text,ids=ids,candidate_ids=cand,source_position=pos))
 byid={r['id']:r for r in rows}
 for r in rows:
  other=byid[r['pair']+('-after' if r['timing']=='before' else '-before')];assert len(r['ids'])==len(other['ids']) and r['source_position']==other['source_position'] and r['ids'][-1]==other['ids'][-1] and r['ids'][r['source_position']]==other['ids'][other['source_position']]
 cfg={'experiment':'H548','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'head':[14,4],'design':'Eight new name-triple combinations x3 role rotations x2 givers xhome/work cue xbefore/after source =192 prompts. Matched distractor token, its absolute position, total length, final token and position within each pair.','modes':['self','zero_distractor','swap_key','swap_query','swap_both'],'hypotheses':['Cue-before lowers harmful distractor reading if source-key availability limits this pathway.','Swapping the before-cue key into the after-cue receiver transfers the effect if key context is sufficient.','Query swaps may instead carry the change, or both operands may matter.','Deletion rescue transfers to these new prompts without fitting.'],'scope':'Before/after clauses express analogous absence from market, but wording and syntax differ. This does not isolate a pure irrelevance concept. Real Q/K swaps affect other edges/destinations of that head. Deletion touches only distractor bus contribution at final destination.','selection':'Head, source role and all conditions frozen before new evaluations; no case filtering.','precision':'float32 compute after bfloat16-rounded loader'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2)+'\n');mh=hashlib.sha256(mp.read_bytes()).hexdigest();print('LOCKED',len(rows),flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 l,h=14,4;H=model.config.n_head;D=model.config.n_embd//H;cache={};native={};buckets=defaultdict(list)
 for r in rows:buckets[len(r['ids'])].append(r)
 for length,rr in sorted(buckets.items()):
  for begin in range(0,len(rr),24):
   batch=rr[begin:begin+24];logits,cc=native_run(model,torch.tensor([r['ids'] for r in batch],device='cuda'),capture=True)
   for ix,r in enumerate(batch):cache[r['id']]={n:v[ix].clone() for n,v in cc.items()};native[r['id']]=score(logits[ix],r['candidate_ids'],r['answer'],r['query'])
 result={'config':cfg,'manifest_sha256':mh,'native':[dict(id=r['id'],item=r['item'],cue=r['cue'],timing=r['timing'],**native[r['id']]) for r in rows],'geometry':[],'interventions':[],'summary':{},'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 jobs=defaultdict(list)
 for r in rows:
  donor=byid[r['pair']+('-after' if r['timing']=='before' else '-before')];T=len(r['ids']);p=r['source_position'];c,dc=cache[r['id']],cache[donor['id']];attn=model.blocks[l].attn;vectors={}
  for origin,ca in [('native',c),('donor',dc)]:
   raw={n:ca[(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vectors[origin]={n:model.tt.apply_rotary_emb(F.rms_norm(v,(D,)),cos,sin)[0,-1 if n in ('c_q','c_q2') else p,h] for n,v in raw.items()}
  for mode in cfg['modes']:
   qo='donor' if mode in ('swap_query','swap_both') else 'native';ko='donor' if mode in ('swap_key','swap_both') else 'native';f1=float(vectors[qo]['c_q']@vectors[ko]['c_k']/D);f2=float(vectors[qo]['c_q2']@vectors[ko]['c_k2']/D);coef=f1*f2*float(attn.lamb);result['geometry'].append(dict(id=r['id'],cue=r['cue'],timing=r['timing'],mode=mode,factor1=f1,factor2=f2,coefficient=coef));qc=dc if qo=='donor' else c;kc=dc if ko=='donor' else c;qq={n:qc[(l,n)][-1,h*D:(h+1)*D] for n in ('c_q','c_q2')};kk={n:kc[(l,n)][p,h*D:(h+1)*D] for n in ('c_k','c_k2')};delta=-coef*F.linear(c['bus'][p,h].cuda(),attn.c_proj.weight[:,h*D:(h+1)*D]) if mode=='zero_distractor' else torch.zeros(model.config.n_embd,device='cuda');jobs[(T,p)].append((r,mode,qq,kk,delta.cpu()))
 save();torch.save({'rows':rows,'cache':cache},ART/f'{STEM}_native_cache.pt')
 for (T,p),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),24):
   batch=jj[begin:begin+24];qq={n:torch.stack([q[n] for r,m,q,k,d in batch]) for n in ('c_q','c_q2')};kk={n:torch.stack([k[n] for r,m,q,k,d in batch]) for n in ('c_k','c_k2')};logits=swapped_run(model,torch.tensor([r['ids'] for r,m,q,k,d in batch],device='cuda'),l,h,p,qq,kk,torch.stack([d for r,m,q,k,d in batch]))
   for ix,(r,mode,q,k,d) in enumerate(batch):ss=score(logits[ix],r['candidate_ids'],r['answer'],r['query']);result['interventions'].append(dict(id=r['id'],item=r['item'],rotation=r['rotation'],cue=r['cue'],timing=r['timing'],mode=mode,**ss,correct_margin_delta=ss['correct_margin']-native[r['id']]['correct_margin'],nonquery_margin_delta=ss['nonquery_margin']-native[r['id']]['nonquery_margin']))
  save()
 for cue in clauses:
  for timing in ('before','after'):
   result['summary'][cue+'_'+timing]={}
   for mode in cfg['modes']:
    cc=[c for c in result['interventions'] if c['cue']==cue and c['timing']==timing and c['mode']==mode];gg=[g for g in result['geometry'] if g['cue']==cue and g['timing']==timing and g['mode']==mode];result['summary'][cue+'_'+timing][mode]={'candidate_accuracy':float(np.mean([c['candidate_correct'] for c in cc])),'full_vocab_top1':float(np.mean([c['full_top1'] for c in cc])),'mean_correct_margin':float(np.mean([c['correct_margin'] for c in cc])),'mean_margin_delta':float(np.mean([c['correct_margin_delta'] for c in cc])),'mean_coefficient':float(np.mean([g['coefficient'] for g in gg]))}
 result['audits']={'self_max_margin_delta':max(abs(c['correct_margin_delta']) for c in result['interventions'] if c['mode']=='self'),'all_96_pairs_match_tokens_positions_lengths':True,'finite':all(np.isfinite(c['correct_margin']) for c in result['interventions']),'manifest_unchanged':mh==hashlib.sha256(mp.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',result['summary'],flush=True)
if __name__=='__main__':main()
