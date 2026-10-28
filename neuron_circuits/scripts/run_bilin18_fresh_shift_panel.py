"""H640 prospective matched-ending versus broad-format panel, frozen before inference."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_fresh_shift_panel'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);words='river mountain window garden bottle pencil paper music'.split();countries='Canada Ireland Egypt Belgium Peru Thailand Denmark Hungary'.split();answers='Ottawa Dublin Cairo Brussels Lima Bangkok Copenhagen Budapest'.split();defs={'copy':(words,words,['Repeat this word: {}.\nAnswer:','Copy this word: {}.\nAnswer:','Repeat the word: {}. The word is','Copy exactly: {}\nCopy:']),'fact':(countries,answers,['Name the capital of {}.\nAnswer:','Give the capital of {}.\nAnswer:','The capital of {} is','What is the capital of {}?\nAnswer:'])};rows=[]
 def token(s):
  ids=tok.encode(' '+s,add_special_tokens=False);assert len(ids)==1,(s,ids);return ids[0]
 old=json.loads((ART/'bilin18_cross_task_response_panel_manifest.json').read_text());old+=json.loads((ART/'bilin18_prospective_operator_panel_manifest.json').read_text());oldprompts={e['prompt'] for e in old}
 for family,(subjects,ans,templates) in defs.items():
  for form,template in enumerate(templates):
   for item,(subject,answer) in enumerate(zip(subjects,ans)):
    prompt=template.format(subject);assert prompt not in oldprompts;rows.append(dict(id=f'{family}-{form}-{item}',family=family,form=form,item=item,contrast='matched' if form<2 else 'broad',subject=subject,answer=answer,answer_id=token(answer),candidate_ids=[token(a) for a in ans],prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False)))
 assert not set(countries)&set('France Germany Italy Japan Spain China Russia England'.split());assert not set(words)&set('cat dog book car tree house bird boy'.split())
 assert not set(words+countries)&{e.get('subject','') for e in old}
 lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};lengths=[len(e['ids'])==len(rows[lookup[e['family'],e['form']^1,e['item']]]['ids']) for e in rows if e['form']<2];assert all(lengths)
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));pp=ART/'bilin18_fresh_shift_probes_locked.pt';allp=torch.load(pp,weights_only=False);probes={n:allp[n] for n in list(allp)};paths={'manifest':mp,'probes':pp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};cfg=dict(experiment='H642-panel',hashes=hashes,design='64 prompts:2 families x8 new items x4 forms. Forms0/1 differ by instruction verb, equal token length and identical answer-introducing suffix; forms2/3 use prior broad-format contrast. All targets single tokens. Freeze before inference; no filtering. Original anchor plus three fresh probes, +/-0.02.',scope='New relative to H633 task panel; some country names appeared in earlier language tasks. Matched templates change one lexical instruction, not every possible construction attribute.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 caches={l:{} for l in range(10,18)};bus={};x0={};forwards=0
 def evaluate(q=None,alpha=0.):
  nonlocal forwards
  zall=torch.empty(len(rows),model.lm_head.weight.shape[0]);caps={l:torch.empty(len(rows),1152) for l in [9,17]}
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hs=[]
    if q is not None:
     def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*q.to(y);return y,b
     hs.append(model.blocks[5].attn.register_forward_hook(edit))
    for l in [9,17]:
     def ch(m,args,out,l=l):caps[l][bb]=out[0][:,-1].cpu()
     hs.append(model.blocks[l].register_forward_hook(ch))
    if q is None:
     for l in range(10,18):
      def pre(m,args,l=l):
       for k,i in enumerate(bb):
        caches[l][i]=args[0][k].cpu().clone()
        if l==10:bus[i]=args[1][k].cpu().clone();x0[i]=args[2][k].cpu().clone()
      hs.append(model.blocks[l].register_forward_pre_hook(pre))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));zall[bb]=z.cpu()
    finally:
     for h in hs:h.remove()
    forwards+=len(bb)
  return zall,caps
 native,nativecaps=evaluate();responses={};tangents={}
 for n,q in probes.items():
  zm,cm=evaluate(q,-.02);zp,cp=evaluate(q,.02);responses[n]=(zp.double()-zm.double())/.04;tangents[n]={l:(cp[l].double()-cm[l].double())/.04 for l in [9,17]};print('CAPTURED',n,flush=True)
 competence=[]
 for i,e in enumerate(rows):
  z=native[i];cs=e['candidate_ids'];competence.append(dict(id=e['id'],family=e['family'],form=e['form'],contrast=e['contrast'],top1_correct=int(z.argmax())==e['answer_id'],candidate_correct=cs[int(z[cs].argmax())]==e['answer_id'],top_token=tok.decode([int(z.argmax())]),target_probability=float(z.double().softmax(0)[e['answer_id']])))
 torch.save(dict(caches=caches,bus=bus,x0=x0,native_logits=native,native_captures=nativecaps,responses=responses,tangents=tangents),ART/f'{STEM}_captures.pt');audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=forwards,matched_token_lengths_equal=all(lengths),prompt_count=len(rows));(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,competence=competence,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
