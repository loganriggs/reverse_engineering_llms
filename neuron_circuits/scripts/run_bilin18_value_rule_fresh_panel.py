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
STEM='bilin18_value_rule_fresh_panel'
def declarations():
 nouns=[('cup','cups'),('room','rooms'),('train','trains'),('flower','flowers')]
 antonyms=[('open','closed'),('full','empty'),('rich','poor'),('clean','dirty')]
 capitals=[('Canada','Ottawa'),('Ireland','Dublin'),('Egypt','Cairo'),('Belgium','Brussels')]
 months=[('September','October'),('October','November'),('November','December'),('December','January')]
 tasks=[]
 for family,pairs,operation in [('plural',nouns,'the plural'),('antonym',antonyms,'the opposite'),('capital',capitals,'the capital'),('successor',months,'the following month')]:
  for i,(x,y) in enumerate(pairs):tasks.append(dict(family=family,item=i,input=x,answer=y,copy_answer=x,operation=operation))
 for i,(a,b) in enumerate([(6,1),(6,2),(6,3),(7,3)]):tasks.append(dict(family='addition',item=i,input=f'{a} + {b}',answer=str(a+b),copy_answer=str(a),operation='the sum'))
 for i,(noun,ans) in enumerate([('cup','is'),('rooms','are'),('train','is'),('flowers','are')]):tasks.append(dict(family='agreement',item=i,input=f'The {noun}',answer=ans,copy_answer='The',operation='the present-tense form of be that agrees with this subject'))
 return tasks
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);tasks=declarations();rows=[]
 paraphrases={'plural':'the plural noun','antonym':'the antonym','capital':'the capital city','successor':'the next month','addition':'the total','agreement':'the correct present-tense conjugation of be for this subject'}
 for t in tasks:t['operation']=paraphrases[t['family']]
 for t in tasks:
  for control in ['task','copy','constant']:
   instruction=({'task':f"{{verb}} {t['operation']} for the input.",'copy':'{verb} the first word or number of the input unchanged.','constant':'{verb} the word yes regardless of the input.'})[control]
   answer={'task':t['answer'],'copy':t['copy_answer'],'constant':'yes'}[control]
   for form,(verb,suffix) in enumerate([('Give','Answer:'),('Provide','Answer:'),('Give','Result:'),('Provide','Result:')]):
    prompt=instruction.format(verb=verb)+f"\nInput: {t['input']}\n{suffix}";target=tok.encode(' '+answer,add_special_tokens=False)
    assert len(target)==1,(t,control,answer,target)
    rows.append(dict(**t,id=f"{t['family']}-{t['item']}-{control}-{form}",control=control,form=form,verb=verb,suffix=suffix,split='prospective',expected=answer,answer_id=target[0],prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False)))
 assert len(rows)==288 and len({r['id'] for r in rows})==288
 for family in {r['family'] for r in rows}:
  for control in ['task','copy','constant']:
   rr=[r for r in rows if r['family']==family and r['control']==control];cs=sorted(set(r['answer_id'] for r in rr))
   for r in rr:r['candidate_ids']=cs
 for t in tasks:
  rr=[r for r in rows if r['family']==t['family'] and r['item']==t['item']];assert len(rr)==12 and all(r['input']==t['input'] for r in rr)
 old=json.loads((ART/'bilin18_operation_paraphrase_panel_manifest.json').read_text());assert not {r['prompt'] for r in rows}&{r['prompt'] for r in old}
 manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));decl=ART/f'{STEM}_declarations.json';decl.write_text(json.dumps(tasks,indent=2));paths={'manifest':manifest,'declarations':decl,'runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 cfg=dict(experiment='H666-panel',hashes=hashes,design='6 families x4 new items x3 controls x4 forms =288 prompts. Frozen before inference; no discovery or fitting here. All targets single tokens and all prompts disjoint from H653/H655. Constant candidate metric trivial.',limitations='New relative to this predictor panel; some words/countries appeared in earlier research. Same templates and model; only next-token competence, not full-response correctness.')
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
