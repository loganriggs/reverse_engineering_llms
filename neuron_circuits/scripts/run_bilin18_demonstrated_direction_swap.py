"""H679: fixed three-way direction/copy panel and coefficient transplants."""
import json,time,hashlib,re
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_demonstrated_direction_swap'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);rows=[];months='January February March April May June July August September October November December'.split()
 def token(s):
  ids=tok.encode(' '+s,add_special_tokens=False);assert len(ids)==1;return ids[0]
 for family in ['month','integer']:
  values=months if family=='month' else [str(k) for k in range(2,14)];demo=[0,3,6,9];test=[i for i in range(12) if i not in demo]
  def ans(k,c):return values[k] if c=='copy' else (months[(k+(1 if c=='next' else -1))%12] if family=='month' else str(int(values[k])+(1 if c=='next' else -1)))
  for control in ['next','previous','copy']:
   instruction='Repeat the first word or number of each input unchanged.' if control=='copy' else f"Give the {'following' if control=='next' else 'preceding'} {family} for each input."
   prefix=instruction+'\n'+'\n'.join(f'{values[k]} -> {ans(k,control)}' for k in demo)+'\n'
   for item,k in enumerate(test):
    prompt=prefix+f'{values[k]} ->';enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);qs=len(prefix);qe=qs+len(values[k]);pos=[n for n,(a,b) in enumerate(enc['offset_mapping']) if max(a,qs)<min(b,qe) and prompt[max(a,qs):min(b,qe)].strip()];assert len(pos)==1
    rows.append(dict(id=f'{family}-{item}-{control}',family=family,control=control,item=item,input=values[k],next=ans(k,'next'),previous=ans(k,'previous'),copy=ans(k,'copy'),expected=ans(k,control),answer_id=token(ans(k,control)),prompt=prompt,ids=enc['input_ids'],query_position=pos[0]))
 assert len(rows)==48;manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));paths={'manifest':manifest,'runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cfg=dict(experiment='H679',hashes=hashes,design='Months and integers,8query inputs each,4disjoint demonstrations, next/previous/copy arrow prompts. Native competence threshold6/8 per operation fixed before inference. All paired donor operations tested, receiver value fixed. All outcomes retained regardless of competence.',limitations='One format/demo order, small reused month vocabulary. Changing operations changes demonstrations as well as instruction. Coefficient transplants are local activation edits, not complete circuit interchange.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 att=model.blocks[14].attn;W=att.c_proj.weight[:,512:640].double();V=torch.empty(48,1152,dtype=torch.float64);coef=torch.empty(48,dtype=torch.float64);buckets=defaultdict(list);lookup={(r['family'],r['item'],r['control']):i for i,r in enumerate(rows)}
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 outputs={};forwards=0;write_errors=[]
 for mode in ['baseline','next','previous','copy']:
  zall=torch.empty(48,model.lm_head.weight.shape[0])
  donor=[lookup[r['family'],r['item'],mode] for r in rows] if mode!='baseline' else list(range(48));dc=coef[donor]-coef if mode!='baseline' else None
  for inds in buckets.values():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8]
    if mode=='baseline':
     def inspect(m,args,out):
      y,bus=args;B,T,_=y.shape;q,k,q2,k2=[mod(y).view(B,T,9,128) for mod in [att.c_q,att.c_k,att.c_q2,att.c_k2]];cos,sin=att.rotary(q);q,k,q2,k2=[model.tt.apply_rotary_emb(F.rms_norm(v,(128,)),cos,sin) for v in [q,k,q2,k2]];c=((q[:,-1,None]*k).sum(-1)/128)*((q2[:,-1,None]*k2).sum(-1)/128);values=(1-att.lamb)*att.c_v(y).view(B,T,9,128)+att.lamb*bus.view(B,T,9,128);pred=(c[...,None].double()*values.double()).sum(1).reshape(B,1152)@att.c_proj.weight.double().T;truth=out[0][:,-1].double();write_errors.extend(((pred-truth).norm(dim=-1)/truth.norm(dim=-1)).cpu().tolist())
      for n,i in enumerate(bb):p=rows[i]['query_position'];coef[i]=c[n,p,4].cpu();V[i]=(values[n,p,4].double()@W.T).cpu()
     hook=att.register_forward_hook(inspect)
    else:
     patch=dc[bb,None]*V[bb]
     def edit(m,args,out):
      y,b=out;y=y.clone();y[:,-1]+=patch.to(y);return y,b
     hook=att.register_forward_hook(edit)
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));zall[bb]=z.cpu()
    finally:hook.remove()
    forwards+=len(bb)
  outputs[mode]=zall;print('COMPLETE',mode,flush=True)
 native=outputs['baseline'].double();lp0=native.log_softmax(-1);p0=lp0.exp();records=[]
 for mode,z in outputs.items():
  lp=z.double().log_softmax(-1)
  for i,r in enumerate(rows):
   ids={c:token(r[c]) for c in ['next','previous','copy']};top=int(z[i].argmax());d=i if mode=='baseline' else lookup[r['family'],r['item'],mode];entry=dict(id=r['id'],family=r['family'],control=r['control'],item=r['item'],mode=mode,coefficient=float(coef[i]),donor_coefficient=float(coef[d]),target_ce_delta=float(lp0[i,r['answer_id']]-lp[i,r['answer_id']]),native_kl=float((p0[i]*(lp0[i]-lp[i])).sum()),top1_correct=top==r['answer_id'],prediction=tok.decode([top]),next_minus_previous_delta=float(z[i,ids['next']]-z[i,ids['previous']]-native[i,ids['next']]+native[i,ids['previous']]))
   for c,tid in ids.items():entry[c+'_top1']=top==tid;entry[c+'_logit_delta']=float(z[i,tid]-native[i,tid])
   records.append(entry)
 replay=max(float((outputs[r['control']][i]-outputs['baseline'][i]).abs().max()) for i,r in enumerate(rows));assert replay<1e-3 and max(write_errors)<1e-4
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},identity_swap_replay_max_abs=replay,attention_reconstruction_max_relative=max(write_errors),whole_model_forwards=forwards,finite=all(bool(torch.isfinite(v).all()) for v in outputs.values()))
 torch.save(dict(logits=outputs,coefficients=coef,receiver_values=V),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()
