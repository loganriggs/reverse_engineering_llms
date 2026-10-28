"""H653: predeclared diverse tasks crossed with copy and constant-answer controls."""
import json,hashlib,time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_direction_matched_panel'
def declarations():
 months=['January','February','March','April','May','June','July','August','September','October','November','December'];tasks=[]
 for i,x in enumerate(months):tasks.append(dict(family='month',item=i,input=x,next=months[(i+1)%12],previous=months[(i-1)%12],unit='month',split='discovery' if i in [0,3,6,9] else 'transfer'))
 for i,n in enumerate(range(2,10)):tasks.append(dict(family='number',item=i,input=str(n),next=str(n+1),previous=str(n-1),unit='number',split='discovery' if i in [0,2,4,6] else 'transfer'))
 return tasks
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);tasks=declarations();rows=[]
 for t in tasks:
  for control in ['next','previous','copy']:
   instruction='{verb} the first word or number of the input unchanged.' if control=='copy' else '{verb} the '+control+' '+t['unit']+' for the input.'
   answer=t['input'] if control=='copy' else t[control];aid=tok.encode(' '+answer,add_special_tokens=False);assert len(aid)==1
   for form,(verb,suffix) in enumerate([('Give','Answer:'),('Provide','Answer:'),('Give','Result:'),('Provide','Result:')]):
    prompt=instruction.format(verb=verb)+'\nInput: '+t['input']+'\n'+suffix;rows.append(dict(**t,id=f"{t['family']}-{t['item']}-{control}-{form}",control=control,form=form,verb=verb,suffix=suffix,expected=answer,answer_id=aid[0],prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False)))
 assert len(rows)==240 and len({r['id'] for r in rows})==240
 for family in ['month','number']:
  cs=sorted({r['answer_id'] for r in rows if r['family']==family})
  for r in rows:
   if r['family']==family:r['candidate_ids']=cs
 lookup={(r['family'],r['item'],r['control'],r['form']):r for r in rows}
 for t in tasks:
  for form in range(4):
   a,b=[lookup[t['family'],t['item'],c,form] for c in ['next','previous']];assert len(a['ids'])==len(b['ids']);assert sum(x!=y for x,y in zip(a['ids'],b['ids']))==1;assert a['input']==b['input'] and a['expected']!=b['expected']
 manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));decl=ART/f'{STEM}_declarations.json';decl.write_text(json.dumps(tasks,indent=2));paths={'manifest':manifest,'declarations':decl,'runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 cfg=dict(experiment='H668',hashes=hashes,design='12 months and8 numbers x next/previous/copy x4 forms =240 prompts. Next/previous matched token length and exactly one changed instruction token. Same inputs, candidate sets shared per domain. Four spaced discovery items per domain, remaining items transfer. Explicit cyclic month and arithmetic number answers frozen before inference.',limitations='Familiar inputs and repeated next-month templates, new controlled operation comparison; not entirely unseen-data replication. Single-token next-token accuracy only, shared candidate accuracy remains weaker than generation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 logits=torch.empty(len(rows),model.lm_head.weight.shape[0]);states={l:torch.empty(len(rows),1152) for l in [9,17]};ys={l:torch.empty(len(rows),1152) for l in range(10,18)};caches={l:{} for l in range(10,18)};bus={};x0={};forwards=0
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];hs=[]
   for l in [9,17]:
    def ch(m,args,out,l=l):states[l][bb]=out[0][:,-1].cpu()
    hs.append(model.blocks[l].register_forward_hook(ch))
   for l in range(10,18):
    def mh(m,args,l=l):ys[l][bb]=args[0][:,-1].cpu()
    def pre(m,args,l=l):
     for k,i in enumerate(bb):
      caches[l][i]=args[0][k].cpu().clone()
      if l==10:bus[i]=args[1][k].cpu().clone();x0[i]=args[2][k].cpu().clone()
    hs.append(model.blocks[l].mlp.register_forward_pre_hook(mh));hs.append(model.blocks[l].register_forward_pre_hook(pre))
   try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
 competence=[]
 for i,r in enumerate(rows):
  z=logits[i];cs=r['candidate_ids'];winner=cs[int(z[cs].argmax())];competence.append(dict(id=r['id'],family=r['family'],control=r['control'],form=r['form'],split=r['split'],candidate_correct=winner==r['answer_id'],top1_correct=int(z.argmax())==r['answer_id'],top_token=tok.decode([int(z.argmax())]),candidate_token=tok.decode([winner]),target_probability=float(z.double().softmax(0)[r['answer_id']])))
 torch.save(dict(native_logits=logits,native_captures=states,normalized_mlp_contexts=ys,caches=caches,bus=bus,x0=x0),ART/f'{STEM}_captures.pt');audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,prompt_count=len(rows),all_targets_single_token=True,explicit_item_pairing_verified=True,finite=bool(torch.isfinite(logits).all()))
 (ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,competence=competence,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
